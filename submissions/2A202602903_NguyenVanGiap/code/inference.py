"""Validation-only selection of TTA/calibration; numpy outputs follow loader order."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import minimize_scalar


@torch.inference_mode()
def predict_logits(model, loader, device, view=None):
    model.to(device).eval()
    names, labels, logits = [], [], []
    for x, y, filenames in loader:
        x = x.to(device)
        z = model(x if view is None else view(x))
        names.extend(filenames)
        labels.append(y.numpy())
        logits.append(z.float().cpu().numpy())
    return names, np.concatenate(labels), np.concatenate(logits)


def view_identity(x):
    return x


def view_hflip(x):
    return x.flip(-1)


def views_multicrop(x, crop):
    h, w = x.shape[-2:]
    if not 0 < crop <= min(h, w):
        raise ValueError('crop must fit input')
    return [x[..., y:y+crop, a:a+crop] for y, a in
            [(0, 0), (0, w-crop), (h-crop, 0), (h-crop, w-crop), ((h-crop)//2, (w-crop)//2)]]


def views_multiscale(x, sizes):
    return [F.interpolate(x, size=(s, s), mode='bilinear', align_corners=False, antialias=True) for s in sizes]


def apply_temperature(logits, T):
    if not np.isfinite(T) or T <= 0:
        raise ValueError('T must be finite and positive')
    z = np.asarray(logits, dtype=np.float64) / T
    z -= z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=-1, keepdims=True)


def aggregate_views(logits_per_view, space='prob'):
    z = np.stack(logits_per_view)
    if space == 'prob':
        return apply_temperature(z, 1.).mean(0)
    if space == 'logit':
        return apply_temperature(z.mean(0), 1.)
    raise ValueError('space must be prob/logit')


def ensemble_probs(list_of_probs):
    p = np.stack(list_of_probs).mean(0)
    if (p < 0).any() or not np.isfinite(p).all() or (p.sum(1) <= 0).any():
        raise ValueError('Invalid probabilities')
    return p / p.sum(1, keepdims=True)


def fit_temperature(val_logits, val_labels):
    from scipy.special import logsumexp
    z = np.asarray(val_logits, dtype=np.float64)
    y = np.asarray(val_labels, dtype=int)
    def objective(log_t):
        scaled = z / np.exp(log_t)
        return float((logsumexp(scaled, axis=1) - scaled[np.arange(len(y)), y]).mean())
    result = minimize_scalar(objective, bounds=(-5., 5.), method='bounded')
    return float(np.exp(result.x)) if objective(result.x) <= objective(0.) else 1.


def fuse_conv_bn(model):
    # Fuse only proven forward-adjacent pairs: Sequential or torch.fx graph edges.
    # Merely adjacent registration order can be incorrect in residual blocks.
    from torch.fx import symbolic_trace
    from torch.nn.utils import fuse_conv_bn_eval
    result = copy.deepcopy(model).eval()
    if not any(isinstance(m, nn.BatchNorm2d) for m in result.modules()):
        return result
    graph = symbolic_trace(result)
    calls = {}
    for node in graph.graph.nodes:
        if node.op == 'call_module':
            calls[node.target] = calls.get(node.target, 0) + 1
    for node in graph.graph.nodes:
        if node.op != 'call_module':
            continue
        bn = result.get_submodule(node.target)
        if not isinstance(bn, nn.BatchNorm2d) or not node.args:
            continue
        prev = node.args[0]
        if not hasattr(prev, 'op') or prev.op != 'call_module' or len(prev.users) != 1:
            continue
        conv = result.get_submodule(prev.target)
        if calls.get(prev.target) != 1 or calls.get(node.target) != 1 or not isinstance(conv, nn.Conv2d):
            continue
        result.set_submodule(prev.target, fuse_conv_bn_eval(conv, bn))
        result.set_submodule(node.target, nn.Identity())
    return result
