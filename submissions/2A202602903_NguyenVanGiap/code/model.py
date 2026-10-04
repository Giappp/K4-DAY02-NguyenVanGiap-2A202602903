"""timm classifiers, frozen backbone, optimizer groups and fvcore MAC counts."""
import torch
import timm

SUGGESTED_BACKBONES = {'resnet50': 'resnet50', 'resnext50': 'resnext50_32x4d',
    'convnext_tiny': 'convnext_tiny', 'deit_small': 'deit_small_patch16_224',
    'swin_tiny': 'swin_tiny_patch4_window7_224', 'efficientnet_b0': 'efficientnet_b0',
    'mobilenetv3': 'mobilenetv3_large_100'}


def build_model(name, pretrained=True, num_classes=9, drop_rate=0., init='finetune'):
    if init not in ('scratch', 'frozen', 'finetune'):
        raise ValueError(f'Unknown init: {init}')
    model = timm.create_model(SUGGESTED_BACKBONES.get(name, name), pretrained=pretrained and init != 'scratch',
                              num_classes=num_classes, drop_rate=drop_rate)
    if init == 'frozen':
        freeze_backbone(model)
    return model


def freeze_backbone(model):
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.get_classifier().parameters():
        p.requires_grad_(True)
    model.eval()


def param_groups(model, lr_backbone, lr_head, weight_decay):
    head = {id(p) for p in model.get_classifier().parameters()}
    groups = {}
    # Four groups: head bias/norm also excluded from weight decay (rubric D/H).
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        key = (lr_head if id(p) in head else lr_backbone,
               0. if p.ndim <= 1 or name.endswith('.bias') else weight_decay)
        groups.setdefault(key, []).append(p)
    return [{'params': ps, 'lr': lr, 'weight_decay': wd} for (lr, wd), ps in groups.items()]


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size=224):
    from fvcore.nn import FlopCountAnalysis
    training = model.training
    model.eval()
    try:
        x = torch.zeros(1, 3, img_size, img_size, device=next(model.parameters()).device)
        analysis = FlopCountAnalysis(model, x)
        # fvcore counts one fused multiply-add as one operation; unsupported ops are reported.
        return analysis.total() / 1e9
    finally:
        model.train(training)
