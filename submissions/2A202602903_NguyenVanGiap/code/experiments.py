"""Kaggle experiment orchestration. All selection is based on validation."""
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt
import dataset
import model as models
from train import Config, run, run_dir, set_seed, json_write
from inference import predict_logits, view_hflip, aggregate_views, apply_temperature, fit_temperature
from benchmark import bench
from eval import compute_metrics, read_pred


def eda(cfg):
    frames = dataset.load_split(cfg.labels_dir)
    stats = dataset.check_split(*frames, cfg.images_dir)
    out = Path(cfg.curves_dir); out.mkdir(parents=True, exist_ok=True)
    distribution = pd.DataFrame(stats['per_class']).reindex(range(9))
    distribution.index = dataset.CLASS_NAMES
    distribution.plot.bar(figsize=(12, 5), title='DeepWeeds fold 0: class distribution')
    plt.ylabel('Images'); plt.tight_layout(); plt.savefig(out / 'EDA_distribution.png', dpi=160); plt.show()
    fig, axes = plt.subplots(9, 3, figsize=(10, 24))
    for label in range(9):
        for ax, (_, row) in zip(axes[label], frames[0][frames[0].Label == label].head(3).iterrows()):
            with Image.open(Path(cfg.images_dir) / row.Filename) as image:
                ax.imshow(image)
            ax.set_title(dataset.CLASS_NAMES[label]); ax.axis('off')
    fig.tight_layout(); fig.savefig(out / 'EDA_examples.png', dpi=120); plt.show()
    print('Paper reference: Negatives=9106; each positive class=1009–1125 (reference, not model results).')
    print('Total by class:', sum((f.Label.value_counts().reindex(range(9), fill_value=0) for f in frames)))
    json_write(out / 'split_checks.json', stats)
    return stats


def pipeline_checks(cfg):
    from losses import mix_batch
    set_seed(cfg.seed)
    df = dataset.load_split(cfg.labels_dir)[0].groupby('Label', group_keys=False).head(1)
    loader = dataset.make_loader(df, cfg.images_dir, dataset.build_transforms(False, 224), 9, False, num_workers=0)
    x, y, _ = next(iter(loader))
    device = 'cuda'
    model = models.build_model('resnet18').to(device)
    initial = torch.nn.functional.cross_entropy(model.eval()(x.to(device)), y.to(device)).item()
    print(f'Initial real head loss={initial:.4f}; uniform theoretical reference ln(9)={np.log(9):.4f}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    curve = []
    # Fixed images, no augmentation; eval mode prevents BN statistics from drifting.
    model.eval()
    for _ in range(120):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x.to(device)), y.to(device))
        loss.backward(); optimizer.step(); curve.append(loss.item())
        if loss.item() < .01:
            break
    plt.plot(curve); plt.xlabel('Optimization step'); plt.ylabel('Fixed batch CE'); plt.title('Pipeline overfit check')
    plt.savefig(Path(cfg.curves_dir) / 'pipeline_overfit.png', dpi=160); plt.show()
    print('Final fixed-batch loss:', curve[-1])
    if curve[-1] >= .05:
        raise RuntimeError('Overfit check did not converge; inspect pipeline before full experiments')
    augmented = dataset.DeepWeedsDataset(df, cfg.images_dir, dataset.build_transforms(True, 224, 'color'))
    images = torch.stack([augmented[i][0] for i in range(len(df))])
    mixed, (_, _, lam) = mix_batch(images, y, mode='cutmix')
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for row, batch in zip(axes, (images, mixed)):
        for i, ax in enumerate(row):
            image = batch[i] * torch.tensor(dataset.IMAGENET_STD)[:, None, None] + torch.tensor(dataset.IMAGENET_MEAN)[:, None, None]
            ax.imshow(image.permute(1, 2, 0).clamp(0, 1)); ax.axis('off')
            ax.set_title(dataset.CLASS_NAMES[int(y[i])])
    fig.suptitle(f'Color augmentation (top), CutMix (bottom): lambda={lam:.3f}')
    fig.savefig(Path(cfg.curves_dir) / 'pipeline_augmentation.png', dpi=160); plt.show()
    del model, optimizer
    torch.cuda.empty_cache()


def compare_backbones(base):
    names = ['resnet50', 'resnext50_32x4d', 'convnext_tiny', 'deit_small_patch16_224', 'mobilenetv3_large_100']
    return [run(replace(base, exp_id=f'B{i:02}', backbone=name)) for i, name in enumerate(names, 1)]


def training_sweep(base):
    baseline = run(replace(base, exp_id='T00'))
    # Three axes, baseline value plus one alternative each, with controlled one-factor changes.
    options = [('A', 'init=frozen', {'init': 'frozen'}),
               ('B', 'aug=color', {'aug': 'color'}),
               ('C', 'loss=ls smoothing=0.1', {'loss': 'ls', 'label_smoothing': .1}),
               ('C', 'loss=focal gamma=2', {'loss': 'focal', 'focal_gamma': 2.})]
    rows = [{**baseline, 'axis': 'baseline', 'change': 'T00', 'delta': 0.}]
    gains = {}
    for i, (axis, description, values) in enumerate(options, 1):
        result = run(replace(base, exp_id=f'T{i:02}', **values))
        rows.append({**result, 'axis': axis, 'change': description, 'delta': result['macro_f1_val']-baseline['macro_f1_val']})
        if result['macro_f1_val'] > baseline['macro_f1_val'] and (axis not in gains or result['macro_f1_val'] > gains[axis][0]):
            gains[axis] = (result['macro_f1_val'], values)
    changes = {k: v for _, values in gains.values() for k, v in values.items()}
    combination = run(replace(base, exp_id='T05', **changes))
    rows.append({**combination, 'axis': 'combination', 'change': str(changes),
                 'delta': combination['macro_f1_val']-baseline['macro_f1_val']})
    return rows


def load_run(result):
    cfg = Config(**result['config'])
    model = models.build_model(cfg.backbone, pretrained=False, init='scratch', drop_rate=cfg.drop_rate).cuda().eval()
    checkpoint = torch.load(run_dir(cfg) / 'best.pt', map_location='cuda', weights_only=True)
    model.load_state_dict(checkpoint['model'])
    return cfg, model


def inference_sweep(result):
    cfg = Config(**result['config'])
    path = Path(cfg.out_dir).parent / 'inference_results.json'
    if path.exists():
        cached = json.loads(path.read_text())
        if all(row['checkpoint'] == str(run_dir(cfg) / 'best.pt') for row in cached):
            return cached
        raise ValueError('Inference selection checkpoint changed; use a new session')
    cfg, model = load_run(result)
    val_df = dataset.load_split(cfg.labels_dir)[1]
    loader = dataset.make_loader(val_df, cfg.images_dir, dataset.build_transforms(False, cfg.img_size),
                                 cfg.batch_size, False, num_workers=cfg.num_workers)
    names, y, z = predict_logits(model, loader, 'cuda')
    n2, y2, flipped = predict_logits(model, loader, 'cuda', view_hflip)
    assert names == n2 and np.array_equal(y, y2)
    temperature = fit_temperature(z, y)
    from inference import fuse_conv_bn
    fused = fuse_conv_bn(model).cuda().eval()
    x = torch.randn(1, 3, cfg.img_size, cfg.img_size, device='cuda')
    rows = []
    settings = [('identity', apply_temperature(z, 1.), 'fp32', model),
                ('hflip_prob', aggregate_views([z, flipped], 'prob'), 'fp32', model),
                ('hflip_logit', aggregate_views([z, flipped], 'logit'), 'fp32', model),
                ('temperature', apply_temperature(z, temperature), 'fp32', model),
                ('fused_bn', apply_temperature(z, 1.), 'fp32', fused),
                ('amp', None, 'amp', model)]
    with torch.inference_mode():
        error = (model(x)-fused(x)).abs().max().item()
    if error > 1e-4:
        raise RuntimeError(f'BN fusion error {error}')
    print('BN fusion max error:', error)
    for i, (method, p, dtype, net) in enumerate(settings):
        if method in ('amp', 'fused_bn'):
            # Measure actual outputs rather than assigning FP32 accuracy to AMP/fused model.
            outputs = []
            with torch.inference_mode(), torch.autocast('cuda', enabled=dtype == 'amp'):
                for images, _, _ in loader:
                    outputs.append(net(images.cuda()).float().cpu().numpy())
            p = apply_temperature(np.concatenate(outputs), 1.)
        with torch.inference_mode(), torch.autocast('cuda', enabled=dtype == 'amp'):
            def forward():
                a = net(x)
                if method.startswith('hflip'):
                    b = net(x.flip(-1))
                    return ((a.softmax(1)+b.softmax(1))/2 if method.endswith('prob') else ((a+b)/2).softmax(1))
                return (a / temperature).softmax(1) if method == 'temperature' else a.softmax(1)
            latency = bench(forward, sync=torch.cuda.synchronize)
        metrics = compute_metrics(y, p.argmax(1), p)
        rows.append({'exp_id': f'I{i:02}', 'method': method, 'checkpoint': str(run_dir(cfg) / 'best.pt'),
            'K': 2 if method.startswith('hflip') else 1, 'macro_f1_val': metrics['macro_f1'], 'top1_val': metrics['top1'],
            'ece_val': metrics['ece'], 'nll_val': metrics['nll'], 'temperature': temperature if method == 'temperature' else 1.,
            'dtype': dtype, 'fused_bn': method == 'fused_bn', **latency,
            'images_per_s': 1000/latency['p50'], 'relative_cost': latency['p50']/rows[0]['p50'] if rows else 1.,
            'gpu': torch.cuda.get_device_name(), 'batch': 1, 'img_size': cfg.img_size, 'preprocessing': False})
    path = Path(cfg.out_dir).parent / 'inference_results.json'
    json_write(path, rows)
    del model, fused
    torch.cuda.empty_cache()
    return rows


def final_runs(base, selected, method):
    if method not in ('identity', 'hflip_prob', 'hflip_logit', 'temperature'):
        raise ValueError('Final supported methods: identity/hflip_prob/hflip_logit/temperature')
    final = Config(**selected['config'])
    rows = []
    for seed in (0, 1, 2):
        rows.append(run(replace(base, exp_id='T00', seed=seed, save_test_predictions=True)))
        rows.append(run(replace(final, exp_id='F01', seed=seed, inference_method=method, save_test_predictions=True)))
    return rows


def export_results(base, backbones, training, inference, final):
    output = Path(base.out_dir).parent
    def flatten(rows):
        return pd.DataFrame([{**{k: v for k, v in r.items() if k not in ('config', 'latency')},
                              **r.get('latency', {})} for r in rows])
    b, t, i, f = map(flatten, (backbones, training, inference, final))
    per_class = []
    curves = Path(base.curves_dir)
    for result in final:
        path = Path(base.pred_dir) / f'{result["exp_id"]}_seed{result["seed"]}_test.csv'
        pred = read_pred(str(path)); m = compute_metrics(pred.y_true, pred.y_pred, pred.probs)
        for label, name in enumerate(dataset.CLASS_NAMES):
            per_class.append({'exp_id': result['exp_id'], 'seed': result['seed'], 'class': name,
                'support': int(m['support'][label]), **{key: m[key][label] for key in ('precision', 'recall', 'f1')}})
        if result['seed'] == 0:
            fig, ax = plt.subplots(figsize=(9, 8))
            im = ax.imshow(m['confusion']); fig.colorbar(im, ax=ax)
            ax.set_xticks(range(9), dataset.CLASS_NAMES, rotation=70); ax.set_yticks(range(9), dataset.CLASS_NAMES)
            ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(f'{result["exp_id"]} seed 0 test confusion')
            fig.tight_layout(); fig.savefig(curves / f'{result["exp_id"]}_confusion.png', dpi=160); plt.close(fig)
            wrong = np.flatnonzero(pred.y_true != pred.y_pred)[:12]
            if len(wrong):
                fig, axes = plt.subplots(3, 4, figsize=(14, 10))
                for ax in axes.flat:
                    ax.axis('off')
                for ax, index in zip(axes.flat, wrong):
                    with Image.open(Path(base.images_dir) / pred.filenames[index]) as image:
                        ax.imshow(image)
                    ax.set_title(f'{dataset.CLASS_NAMES[pred.y_true[index]]}\n→ {dataset.CLASS_NAMES[pred.y_pred[index]]}', fontsize=9)
                fig.tight_layout(); fig.savefig(curves / f'{result["exp_id"]}_errors.png', dpi=140); plt.close(fig)
    aggregate = f.groupby('exp_id')[['macro_f1_val', 'macro_f1_test', 'top1_test', 'ece_test']].agg(['mean', 'std']) if not f.empty else pd.DataFrame()
    if not aggregate.empty:
        aggregate.columns = ['_'.join(c) for c in aggregate.columns]
        aggregate = aggregate.reset_index()
    summaries = pd.concat([b, t], ignore_index=True).sort_values('macro_f1_val', ascending=False).head(10)
    latency = pd.concat([b, t, i, f], ignore_index=True)
    sheets = {'Backbones': b, 'Training': t, 'Inference': i, 'Final': pd.concat([f, aggregate], ignore_index=True),
              'PerClass': pd.DataFrame(per_class), 'Latency': latency, 'Summary': summaries}
    from openpyxl.styles import PatternFill
    with pd.ExcelWriter(output / 'results.xlsx', engine='openpyxl') as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]; ws.freeze_panes = 'A2'; ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.fill = PatternFill('solid', fgColor='DDEBF7')
            for column in ws.columns:
                ws.column_dimensions[column[0].column_letter].width = min(45, max(14, max(len(str(c.value or '')) for c in column[:20])+2))
                for cell in column[1:]:
                    if isinstance(cell.value, float):
                        cell.number_format = '0.0000'
    for result in backbones + training + final:
        expected = curves / f'{result["exp_id"]}_seed{result["seed"]}.png'
        if not expected.exists():
            raise FileNotFoundError(expected)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(i.p95, i.macro_f1_val)
    for _, row in i.iterrows():
        ax.annotate(row.method, (row.p95, row.macro_f1_val), fontsize=8)
    ax.set_xlabel('Batch 1 p95 latency (ms)'); ax.set_ylabel('Validation macro-F1')
    fig.tight_layout(); fig.savefig(curves / 'inference_tradeoff.png', dpi=160); plt.close(fig)
    def markdown(df):
        # No tabulate dependency required.
        return '| ' + ' | '.join(map(str, df.columns)) + ' |\n|' + '|'.join(['---']*len(df.columns)) + '|\n' + '\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in df.itertuples(index=False, name=None))
    report = '# DeepWeeds — kết quả chạy trên Kaggle\n\n'
    report += 'Báo cáo tự động từ lần chạy thật; các giả thuyết về ảnh lỗi cần người làm lab bổ sung sau khi xem ảnh.\n\n'
    test_status = 'Test ở chung kết, 3 seed (0, 1, 2).' if final else 'SMOKE: chưa chạy chung kết/test; số liệu không dùng để nộp bài.'
    report += '## Dữ liệu và thiết lập\nFold 0 nguyên bản, 17.509 ảnh, chọn checkpoint/cấu hình trên validation. ' + test_status + ' Môi trường và tag trọng số: `runs/*/seed*/environment.json`; cấu hình: `config.json`. Chia ngẫu nhiên không theo địa điểm có thể làm kết quả lạc quan.\n\n'
    report += '## Backbone\n' + markdown(b[['exp_id', 'backbone', 'macro_f1_val', 'params_M', 'GMAC', 'p95']]) + '\n\n'
    report += '## Công thức huấn luyện\n' + markdown(t[['exp_id', 'axis', 'change', 'macro_f1_val', 'delta']]) + '\n\n'
    report += 'Vòng sàng lọc dùng một seed; chênh lệch nhỏ chưa đủ chứng minh cải thiện.\n\n'
    report += '## Suy luận\n' + markdown(i[['exp_id', 'method', 'macro_f1_val', 'ece_val', 'p95']]) + '\n\n'
    report += '## Chung kết (mean và sample std qua seed)\n' + markdown(aggregate) + '\n\n'
    if not aggregate.empty:
        indexed = aggregate.set_index('exp_id')
        delta = indexed.loc['F01', 'macro_f1_test_mean']-indexed.loc['T00', 'macro_f1_test_mean']
        noise = indexed.macro_f1_test_std.max()
        report += f'Δ macro-F1 test = {delta:.6f}; std lớn hơn = {noise:.6f}. ' + ('Δ vượt std.' if delta > noise else 'Chưa chứng minh cải thiện vượt nhiễu seed.') + '\n\n'
    report += '## Phân tích lỗi\nXem `curves/F01_confusion.png`, `curves/F01_errors.png` và sheet PerClass, đặc biệt Chinee Apple / Snake Weed. Hình lỗi chỉ dùng sau khi chốt cấu hình.\n\n'
    if (output / 'migration.json').exists():
        report += '## Chuyển phiên\nTiếp tục checkpoint từ phiên trước trên Kaggle. Metadata gốc và phần cứng lưu ở `migration.json`/`migration_originals/`. Độ trễ từng dòng thuộc GPU đã khai báo ở dòng đó; không coi các phép đo trên GPU khác nhau là cùng điều kiện.\n\n'
    report += '## Hạn chế\nMột fold, ít seed, chưa kiểm chứng trên địa điểm/mùa khác. Độ trễ chỉ gồm forward và hậu xử lý suy luận, không gồm đọc ảnh. Đánh giá lại miền triển khai trước khi đưa lên robot.\n'
    (output / 'report.md').write_text(report, encoding='utf-8')
    json_write(output / 'all_results.json', {'backbones': backbones, 'training': training, 'inference': inference, 'final': final})
    return output / 'results.xlsx'
