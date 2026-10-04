"""Run in Colab: pytest submissions/2A202602903_NguyenVanGiap/tests -q."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from losses import FocalLoss, LabelSmoothingCE, class_weights, mix_batch, mixed_loss
from inference import fuse_conv_bn, aggregate_views, apply_temperature, fit_temperature
from train import Config, EMA, parse_overrides, build_scheduler
from model import param_groups


def test_focal_and_smoothing_reduce_to_ce():
    torch.manual_seed(5)
    z, y = torch.randn(16, 9), torch.randint(9, (16,))
    ce = nn.functional.cross_entropy(z, y)
    torch.testing.assert_close(FocalLoss(gamma=0)(z, y), ce, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(LabelSmoothingCE(0)(z, y), ce)


def test_cutmix_actual_area_and_input_unchanged(monkeypatch):
    monkeypatch.setattr(torch, 'randperm', lambda n, device: torch.tensor([1, 0], device=device))
    for seed in range(10):
        np.random.seed(seed)
        x = torch.stack([torch.zeros(3, 11, 13), torch.ones(3, 11, 13)])
        y = torch.tensor([0, 1])
        mixed, (a, b, lam) = mix_batch(x, y)
        assert abs(mixed[0].mean().item() - (1-lam)) < 1e-6
        torch.testing.assert_close(x[0], torch.zeros_like(x[0]))
        assert torch.equal(b, y.flip(0))
        z = torch.randn(2, 9)
        expected = lam * nn.functional.cross_entropy(z, a) + (1-lam)*nn.functional.cross_entropy(z, b)
        torch.testing.assert_close(mixed_loss(nn.CrossEntropyLoss(), z, (a, b, lam)), expected)


def test_fusion_preserves_output_and_source():
    model = nn.Sequential(nn.Conv2d(3, 4, 3, bias=False), nn.BatchNorm2d(4), nn.ReLU()).eval()
    x = torch.randn(2, 3, 16, 16)
    fused = fuse_conv_bn(model)
    torch.testing.assert_close(model(x), fused(x), atol=1e-5, rtol=1e-5)
    assert isinstance(model[1], nn.BatchNorm2d)
    assert isinstance(fused[1], nn.Identity)


def test_fusion_does_not_fuse_branched_conv():
    class Branched(nn.Module):
        def __init__(self):
            super().__init__(); self.conv = nn.Conv2d(3, 3, 1); self.bn = nn.BatchNorm2d(3)
        def forward(self, x):
            y = self.conv(x)
            return self.bn(y) + y
    model = Branched().eval(); fused = fuse_conv_bn(model)
    x = torch.randn(2, 3, 8, 8)
    torch.testing.assert_close(model(x), fused(x))
    assert isinstance(fused.bn, nn.BatchNorm2d)


def test_calibration_and_aggregation():
    rng = np.random.default_rng(4)
    z = rng.normal(size=(100, 9))*8
    y = rng.integers(9, size=100)
    t = fit_temperature(z, y)
    p = apply_temperature(z, t)
    assert t > 0
    np.testing.assert_allclose(p.sum(1), 1)
    np.testing.assert_array_equal(p.argmax(1), z.argmax(1))
    def nll(p):
        return -np.log(p[np.arange(len(y)), y]).mean()
    assert nll(p) <= nll(apply_temperature(z, 1)) + 1e-7
    for space in ('prob', 'logit'):
        np.testing.assert_allclose(aggregate_views([z, z], space), apply_temperature(z, 1))


def test_ema_and_parameter_groups():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__(); self.body = nn.Linear(3, 4); self.head = nn.Linear(4, 9)
        def get_classifier(self):
            return self.head
    model = Tiny(); ema = EMA(model, .5)
    old = ema.model.body.weight.clone()
    with torch.no_grad():
        model.body.weight.add_(2)
    ema.update(model)
    torch.testing.assert_close(ema.model.body.weight, old+1)
    groups = param_groups(model, 1e-4, 1e-3, .05)
    ps = [p for group in groups for p in group['params']]
    assert len(ps) == len({id(p) for p in ps}) == len(list(model.parameters()))
    for group in groups:
        for p in group['params']:
            if p.ndim == 1:
                assert group['weight_decay'] == 0


def test_typed_config():
    assert parse_overrides(['amp=false', 'seed=2', 'ema_decay=0.99', 'sampler=none']) == {
        'amp': False, 'seed': 2, 'ema_decay': .99, 'sampler': None}
    with pytest.raises(ValueError):
        parse_overrides(['unknown=1'])
    with pytest.raises(ValueError):
        parse_overrides(['amp=maybe'])
    assert not Config().save_test_predictions


def test_warmup_then_decay():
    p = nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=1)
    sched = build_scheduler(opt, Config(epochs=4, warmup_epochs=1), 10)
    rates = []
    for _ in range(40):
        rates.append(opt.param_groups[0]['lr']); opt.step(); sched.step()
    assert rates[0] < rates[8] <= rates[10]
    assert rates[-1] < rates[20]


def test_dataset_reads_rgb_and_preserves_order(tmp_path):
    import pandas as pd
    from PIL import Image
    from dataset import DeepWeedsDataset, build_transforms, make_loader
    Image.new('L', (32, 32), color=100).save(tmp_path / 'first.jpg')
    Image.new('RGB', (32, 32), color='red').save(tmp_path / 'second.jpg')
    df = pd.DataFrame({'Filename': ['first.jpg', 'second.jpg'], 'Label': [4, 1]}, index=[7, 9])
    ds = DeepWeedsDataset(df, tmp_path, build_transforms(False, 16))
    assert ds[0][0].shape == (3, 16, 16)
    loader = make_loader(df, tmp_path, build_transforms(False, 16), 2, False, num_workers=0)
    _, y, names = next(iter(loader))
    assert list(names) == ['first.jpg', 'second.jpg']
    assert y.tolist() == [4, 1]


def test_weights_normalized_and_missing_class_rejected():
    for beta in (0., .9, .9999):
        weights = class_weights([10, 20, 30, 40, 50, 60, 70, 80, 90], beta)
        assert abs(weights.mean().item()-1) < 1e-6
        assert weights[0] > weights[-1]
    with pytest.raises(ValueError):
        class_weights([0]+[10]*8)


def test_resume_recovers_test_csv_without_model_forward(tmp_path, monkeypatch):
    from dataclasses import asdict
    import train
    cfg = Config(exp_id='F01', out_dir=str(tmp_path / 'runs'), pred_dir=str(tmp_path / 'predictions'), save_test_predictions=True)
    folder = train.run_dir(cfg); folder.mkdir(parents=True)
    summary = {'exp_id': 'F01', 'seed': 0, 'macro_f1_val': .9, 'config': asdict(cfg)}
    train.json_write(folder / 'config.json', asdict(cfg))
    train.json_write(folder / 'summary_ready.json', summary)
    train.save_predictions(train.pred_path(cfg, 'test'), [f'{i}.jpg' for i in range(9)], np.arange(9), np.eye(9))
    def forbidden(*args, **kwargs):
        raise AssertionError('Recovery must never run model/data loading')
    monkeypatch.setattr(train.dataset, 'load_split', forbidden)
    result = train.run(cfg)
    assert result['macro_f1_test'] == 1.
    assert (folder / 'summary.json').exists()
    # A rerun of the validation sweep reads val metrics while retaining final artifacts.
    from dataclasses import replace
    val_result = train.run(replace(cfg, save_test_predictions=False))
    assert val_result['macro_f1_val'] == .9
    assert 'macro_f1_test' not in val_result
