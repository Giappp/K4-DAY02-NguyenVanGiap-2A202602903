"""One training entry point for all experiments; checkpoint selection uses validation only."""
from __future__ import annotations
import argparse
import copy
import json
import math
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import get_type_hints, get_args
import numpy as np
import pandas as pd
import torch
import timm
import dataset
import model as models
import losses
from inference import apply_temperature
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eval import compute_metrics, save_predictions as _save_predictions

@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug ...
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    curves_dir: str = "curves"
    resume: bool = True
    inference_method: str = "identity"  # identity | hflip_prob | hflip_logit | temperature
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model, cfg):
    return torch.optim.AdamW(models.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay))


def build_scheduler(optimizer, cfg, steps_per_epoch):
    total = cfg.epochs * steps_per_epoch
    warm = round(cfg.warmup_epochs * steps_per_epoch)
    if total <= 0 or warm >= total:
        raise ValueError('epochs/loader must be positive, warmup shorter than training')
    def multiplier(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        return .5 * (1 + math.cos(math.pi * min(1., (step - warm) / max(1, total - warm))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


class EMA:
    def __init__(self, model, decay):
        if not 0 <= decay < 1:
            raise ValueError('EMA decay must be in [0,1)')
        self.decay = decay
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        source = model.state_dict()
        for key, value in self.model.state_dict().items():
            # Copy BN buffers; average learned floating parameters only.
            if key in dict(model.named_parameters()) and value.is_floating_point():
                value.mul_(self.decay).add_(source[key], alpha=1-self.decay)
            else:
                value.copy_(source[key])


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg, device, ema=None):
    model.train()
    if cfg.init == 'frozen':
        model.eval()
        model.get_classifier().train()
    total, n = 0., 0
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        targets = y
        if cfg.mix:
            x, targets = losses.mix_batch(x, y, cfg.mix_alpha, cfg.mix)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=cfg.amp and device.type == 'cuda'):
            z = model(x)
            loss = losses.mixed_loss(criterion, z, targets) if cfg.mix else criterion(z, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= previous_scale:
            scheduler.step()
            if ema:
                ema.update(model)
        total += loss.item() * len(y)
        n += len(y)
    return {'train_loss': total / n, 'lr': optimizer.param_groups[0]['lr']}


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()
    names, labels, logits, total = [], [], [], 0.
    for x, y, filenames in loader:
        x, y = x.to(device), y.to(device)
        z = model(x)
        total += criterion(z, y).item() * len(y)
        names.extend(filenames)
        labels.append(y.cpu().numpy())
        logits.append(z.float().cpu().numpy())
    y, z = np.concatenate(labels), np.concatenate(logits)
    return names, y, z, total / len(y)


def plot_curves(history, path, title):
    import matplotlib.pyplot as plt
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(df.epoch, df.train_loss, label='train')
    axes[0].plot(df.epoch, df.val_loss, label='validation')
    axes[0].set_ylabel('Loss')
    axes[1].plot(df.epoch, df.macro_f1, label='macro-F1 validation')
    axes[1].set_ylabel('Macro-F1')
    for ax in axes:
        ax.set_xlabel('Epoch'); ax.legend(); ax.grid(alpha=.2)
    fig.suptitle(title); fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160); plt.close(fig)


def atomic_save(obj, path):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)


def json_write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')
    tmp.replace(path)


def save_predictions(path, filenames, y_true, probs):
    path = Path(path)
    tmp = path.with_suffix('.tmp.csv')
    _save_predictions(tmp, filenames, y_true, probs)
    tmp.replace(path)
    return path


def run(cfg):
    if cfg.epochs < 1 or cfg.fold != 0:
        raise ValueError('Core lab requires fold 0 and epochs >= 1')
    if cfg.inference_method not in ('identity', 'hflip_prob', 'hflip_logit', 'temperature'):
        raise ValueError('Unknown inference_method')
    directory = run_dir(cfg)
    directory.mkdir(parents=True, exist_ok=True)
    config = asdict(cfg)
    config_path = directory / 'config.json'
    previous = json.loads(config_path.read_text()) if config_path.exists() else config
    # Reading validation results after finalization must not undo or repeat test evaluation.
    if previous['save_test_predictions'] and not cfg.save_test_predictions and {**previous, 'save_test_predictions': False} == config:
        cached_path = directory / 'summary.json'
        if cached_path.exists() and cfg.resume:
            cached = json.loads(cached_path.read_text())
            return {**{k: v for k, v in cached.items() if not k.endswith('_test')}, 'config': config}
        raise ValueError('Finalized run is incomplete; resume with save_test_predictions=True first')
    promoted = previous != config and not previous['save_test_predictions'] and cfg.save_test_predictions
    allowed = {**previous, 'save_test_predictions': cfg.save_test_predictions}
    if previous != config and (not promoted or allowed != config):
        raise ValueError(f'Config differs from existing run: {directory}; use a new exp_id')
    json_write(config_path, config)
    summary_path = directory / 'summary.json'
    if summary_path.exists() and cfg.resume:
        cached = json.loads(summary_path.read_text())
        if not cfg.save_test_predictions or 'macro_f1_test' in cached:
            return cached
    if cfg.save_test_predictions and pred_path(cfg, 'test').exists():
        from eval import read_pred
        ready_path = directory / 'summary_ready.json'
        if not ready_path.exists():
            raise FileExistsError('Test CSV exists without recovery metadata; inspect saved files before continuing')
        summary = json.loads(ready_path.read_text())
        pred = read_pred(str(pred_path(cfg, 'test')))
        metrics = compute_metrics(pred.y_true, pred.y_pred, pred.probs)
        summary.update({f'{k}_test': metrics[k] for k in ('macro_f1', 'top1', 'ece')})
        json_write(directory / 'test_metrics.json', {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in metrics.items()})
        if cfg.inference_method == 'temperature':
            cached = np.load(directory / 'test_logits.npz')
            save_predictions(Path(cfg.pred_dir) / f'{cfg.exp_id}uncal_seed{cfg.seed}_test.csv',
                             cached['filenames'], cached['y_true'], apply_temperature(cached['logits'], 1.))
        json_write(summary_path, summary)
        return summary
    set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, cfg.fold)
    json_write(directory / 'split.json', dataset.check_split(train_df, val_df, test_df, cfg.images_dir))
    train_loader = dataset.make_loader(train_df, cfg.images_dir, dataset.build_transforms(True, cfg.img_size, cfg.aug),
                                       cfg.batch_size, True, cfg.sampler, cfg.num_workers)
    def eval_loader(df):
        return dataset.make_loader(df, cfg.images_dir, dataset.build_transforms(False, cfg.img_size),
                                   cfg.batch_size, False, num_workers=cfg.num_workers)
    val_loader = eval_loader(val_df)
    model = models.build_model(cfg.backbone, init=cfg.init, drop_rate=cfg.drop_rate).to(device)
    if tuple(model.pretrained_cfg.get('mean', dataset.IMAGENET_MEAN)) != dataset.IMAGENET_MEAN or tuple(model.pretrained_cfg.get('std', dataset.IMAGENET_STD)) != dataset.IMAGENET_STD:
        raise ValueError('Weight preprocessing differs from the common ImageNet recipe; select a compatible weight tag')
    counts = train_df.Label.value_counts().reindex(range(9), fill_value=0).to_numpy()
    weights = losses.class_weights(counts, cfg.class_weight_beta or 0.).to(device)
    criterion = losses.build_criterion(cfg.loss, smoothing=cfg.label_smoothing, gamma=cfg.focal_gamma,
                                      weight=weights).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp and device.type == 'cuda')
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay is not None else None
    history, best, best_epoch, start = [], -1., 0, 0
    latest_path = directory / 'latest.pt'
    if cfg.resume and latest_path.exists():
        state = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']); scaler.load_state_dict(state['scaler'])
        if ema:
            ema.model.load_state_dict(state['ema'])
        history, best, best_epoch, start = state['history'], state['best'], state['best_epoch'], state['epoch']
        random.setstate(state['random']); np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch_rng'].cpu())
        train_loader.generator.set_state(state['loader_rng'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
    json_write(directory / 'environment.json', {'python': platform.python_version(), 'torch': torch.__version__,
        'torchvision': __import__('torchvision').__version__, 'timm': timm.__version__,
        'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU',
        'pretrained_cfg': model.pretrained_cfg, 'deterministic_cudnn': True})
    for epoch in range(start, cfg.epochs):
        t0 = time.perf_counter()
        training = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        seconds = time.perf_counter() - t0
        selected = ema.model if ema else model
        names, y, z, loss = evaluate(selected, val_loader, criterion, device)
        p = apply_temperature(z, 1.)
        metrics = compute_metrics(y, p.argmax(1), p)
        history.append({'epoch': epoch+1, **training, 'val_loss': loss, 'macro_f1': metrics['macro_f1'],
                        'top1': metrics['top1'], 'train_seconds': seconds})
        if metrics['macro_f1'] > best:
            best, best_epoch = metrics['macro_f1'], epoch+1
            atomic_save({'model': selected.state_dict(), 'epoch': best_epoch}, directory / 'best.pt')
        atomic_save({'epoch': epoch+1, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
            'ema': ema.model.state_dict() if ema else None, 'history': history, 'best': best,
            'best_epoch': best_epoch, 'random': random.getstate(), 'numpy': np.random.get_state(),
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
            'loader_rng': train_loader.generator.get_state()}, latest_path)
        pd.DataFrame(history).to_csv(directory / 'history.csv', index=False)
        plot_curves(history, Path(cfg.curves_dir) / f'{cfg.exp_id}_seed{cfg.seed}.png', f'{cfg.exp_id} {cfg.backbone} seed {cfg.seed}')
        print(f'{cfg.exp_id} seed={cfg.seed} epoch={epoch+1}/{cfg.epochs} val_F1={metrics["macro_f1"]:.4f}', flush=True)
    model.load_state_dict(torch.load(directory / 'best.pt', map_location=device, weights_only=True)['model'])
    from inference import predict_logits, view_hflip, aggregate_views, fit_temperature
    def predictions(loader, split, temperature=1.):
        cached_path = directory / f'{split}_logits.npz'
        if split == 'test' and cached_path.exists():
            with np.load(cached_path) as cached:
                return cached['filenames'].tolist(), cached['y_true'], cached['logits'], cached['probs']
        names, y, z, loss = evaluate(model, loader, criterion, device)
        p = apply_temperature(z, temperature)
        if cfg.inference_method.startswith('hflip'):
            other_names, other_y, flipped = predict_logits(model, loader, device, view_hflip)
            assert names == other_names and np.array_equal(y, other_y)
            p = aggregate_views([z, flipped], 'prob' if cfg.inference_method.endswith('prob') else 'logit')
        cache_path = directory / f'{split}_logits.npz'
        tmp_cache = cache_path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp_cache, filenames=np.array(names), y_true=y, logits=z, probs=p)
        tmp_cache.replace(cache_path)
        return names, y, z, p
    names, y, z, p = predictions(val_loader, 'val')
    temperature = fit_temperature(z, y) if cfg.inference_method == 'temperature' else 1.
    if cfg.inference_method == 'temperature':
        p = apply_temperature(z, temperature)
    save_predictions(pred_path(cfg, 'val'), names, y, p)
    val_metrics = compute_metrics(y, p.argmax(1), p)
    from benchmark import inference_latency_report
    latency = inference_latency_report(model, cfg.img_size, cfg.inference_method, temperature, str(device))
    summary = {'exp_id': cfg.exp_id, 'seed': cfg.seed, 'backbone': cfg.backbone, 'init': cfg.init,
        'weight_tag': 'random' if cfg.init == 'scratch' else model.pretrained_cfg.get('architecture', cfg.backbone) + '.' + model.pretrained_cfg.get('tag', ''),
        'params_M': models.count_params(model), 'GMAC': models.count_gmacs(model, cfg.img_size),
        'gmac_tool': 'fvcore (unsupported operations omitted)', 'img_size': cfg.img_size, 'epochs': cfg.epochs,
        'best_epoch': best_epoch, 'macro_f1_val': val_metrics['macro_f1'], 'top1_val': val_metrics['top1'],
        'ece_val': val_metrics['ece'], 'train_seconds_per_epoch': float(np.mean([h['train_seconds'] for h in history])),
        'temperature': temperature, 'inference_method': cfg.inference_method, 'latency': latency,
        'config': config}
    if cfg.save_test_predictions:
        json_write(directory / 'summary_ready.json', summary)
        names, y, z, p = predictions(eval_loader(test_df), 'test', temperature)
        save_predictions(pred_path(cfg, 'test'), names, y, p)
        if cfg.inference_method == 'temperature':
            save_predictions(Path(cfg.pred_dir) / f'{cfg.exp_id}uncal_seed{cfg.seed}_test.csv', names, y, apply_temperature(z, 1.))
        metrics = compute_metrics(y, p.argmax(1), p)
        summary.update({f'{k}_test': metrics[k] for k in ('macro_f1', 'top1', 'ece')})
        json_write(directory / 'test_metrics.json', {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in metrics.items()})
    json_write(summary_path, summary)
    return summary


def parse_overrides(pairs):
    hints = get_type_hints(Config)
    result = {}
    for pair in pairs:
        if '=' not in pair:
            raise ValueError(f'Expected KEY=VALUE: {pair}')
        key, value = pair.split('=', 1)
        if key not in hints:
            raise ValueError(f'Unknown config field: {key}')
        hint = hints[key]
        args = get_args(hint)
        if value.lower() in ('none', 'null') and type(None) in args:
            result[key] = None
            continue
        typ = next((t for t in args if t is not type(None)), hint)
        if typ is bool:
            if value.lower() not in ('true', 'false', '1', '0'):
                raise ValueError(f'{key}: expected true/false')
            result[key] = value.lower() in ('true', '1')
        else:
            result[key] = typ(value)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', nargs='*', default=[])
    args = parser.parse_args()
    print(json.dumps(run(Config(**parse_overrides(args.set))), indent=2))


if __name__ == '__main__':
    main()
