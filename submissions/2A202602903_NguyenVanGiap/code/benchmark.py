"""Model/TTA latency excludes loading/preprocessing; real synchronized timings."""
import copy
import time
import numpy as np
import torch


def bench(fn, warmup=10, iters=100, sync=None):
    if warmup < 10 or iters < 50:
        raise ValueError('Require warmup >= 10 and iters >= 50')
    sync = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - start) * 1000)
    return dict(zip(('p50', 'p95', 'p99'), map(float, np.percentile(times, [50, 95, 99]))),
                mean=float(np.mean(times)), n=iters)


def _report(model, batch_size=1, img_size=224, dtype='fp32', device='cuda', warmup=10, iters=100, k=1, method=None, temperature=1.):
    if dtype not in ('fp32', 'amp', 'fp16') or k < 1:
        raise ValueError('Invalid dtype or views')
    device = torch.device(device)
    m = copy.deepcopy(model).to(device).eval()
    m = m.half() if dtype == 'fp16' else m.float()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device,
                    dtype=torch.float16 if dtype == 'fp16' else torch.float32)
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == 'cuda' else None
    with torch.inference_mode(), torch.autocast(device.type, enabled=dtype == 'amp'):
        def forward():
            outputs = [m(x if i % 2 == 0 else x.flip(-1)) for i in range(k)]
            if k > 1:
                if method == 'hflip_logit':
                    return torch.stack(outputs).mean(0).softmax(1)
                return torch.stack([z.softmax(1) for z in outputs]).mean(0)
            if method is not None:
                return (outputs[0] / temperature).softmax(1)
            return outputs[0]
        measured = bench(forward, warmup, iters, sync)
    return {**measured, 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
            'dtype': dtype, 'batch': batch_size, 'img_size': img_size, 'K': k,
            'images_per_s': batch_size * 1000 / measured['p50'], 'torch': torch.__version__,
            'preprocessing': False}


def latency_report(model, batch_size, img_size, dtype='fp32', device='cuda', warmup=10, iters=100):
    return _report(model, batch_size, img_size, dtype, device, warmup, iters)


def tta_latency(model, k_views, **kw):
    base = _report(model, **kw)
    result = _report(model, k=k_views, **kw)
    return {**result, 'single_p50': base['p50'], 'expected_p50': k_views * base['p50']}


def inference_latency_report(model, img_size, method='identity', temperature=1., device='cuda'):
    return _report(model, batch_size=1, img_size=img_size, device=device,
                   k=2 if method.startswith('hflip') else 1, method=method, temperature=temperature)
