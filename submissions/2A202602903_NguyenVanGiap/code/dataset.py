"""DeepWeeds: official folds, reproducible loaders and ImageNet transforms."""
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T

NUM_CLASSES = 9
CLASS_NAMES = ['Chinee Apple', 'Lantana', 'Parkinsonia', 'Parthenium', 'Prickly Acacia',
               'Rubber Vine', 'Siam Weed', 'Snake Weed', 'Negatives']
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir, fold=0):
    if fold not in range(5):
        raise ValueError('fold must be 0..4')
    return tuple(pd.read_csv(Path(labels_dir) / f'{s}_subset{fold}.csv') for s in ('train', 'val', 'test'))


def check_split(train_df, val_df, test_df, images_dir):
    frames = dict(zip(('train', 'val', 'test'), (train_df, val_df, test_df)))
    names = {}
    for split, df in frames.items():
        if not {'Filename', 'Label'}.issubset(df.columns):
            raise ValueError(f'{split}: missing columns')
        if df.Filename.isna().any() or df.Filename.duplicated().any():
            raise ValueError(f'{split}: missing/duplicate filenames')
        if not df.Label.isin(range(NUM_CLASSES)).all():
            raise ValueError(f'{split}: invalid labels')
        names[split] = set(df.Filename)
        missing = [f for f in names[split] if not (Path(images_dir) / f).is_file()]
        if missing:
            raise FileNotFoundError(f'{split}: missing {len(missing)} images, e.g. {missing[:3]}')
    overlap = {f'{a}_{b}': len(names[a] & names[b]) for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
    if any(overlap.values()) or len(set.union(*names.values())) != 17509:
        raise ValueError(f'Invalid official split: overlap={overlap}; expected union=17509')
    return {'n': {s: len(d) for s, d in frames.items()}, 'overlap': overlap,
            'per_class': {s: {int(k): int(v) for k, v in d.Label.value_counts().sort_index().items()} for s, d in frames.items()}}


def build_transforms(train, img_size=224, aug='basic'):
    if img_size <= 0:
        raise ValueError('img_size must be positive')
    extras = {'basic': [], 'color': [T.ColorJitter(.2, .2, .2, .05)],
              'trivial': [T.TrivialAugmentWide()], 'randaug': [T.RandAugment(num_ops=2, magnitude=9)]}
    if aug not in extras:
        raise ValueError(f'Unknown augmentation: {aug}')
    # Resize first: higher evaluation resolutions must not silently zero-pad 256px images.
    ops = ([T.RandomResizedCrop(img_size), T.RandomHorizontalFlip(), *extras[aug]] if train
           else [T.Resize(max(256, round(img_size * 256 / 224))), T.CenterCrop(img_size)])
    return T.Compose([*ops, T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class DeepWeedsDataset(Dataset):
    def __init__(self, df, images_dir, transform=None):
        self.df, self.images_dir, self.transform = df.reset_index(drop=True), Path(images_dir), transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        with Image.open(self.images_dir / row.Filename) as image:
            image = image.convert('RGB')
            image = self.transform(image) if self.transform else image.copy()
        return image, int(row.Label), str(row.Filename)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(df, images_dir, transform, batch_size, train, sampler=None, num_workers=2):
    if sampler not in (None, 'balanced') or (sampler and not train):
        raise ValueError('balanced sampler is only supported for training')
    generator = torch.Generator().manual_seed(torch.initial_seed())
    sample = None
    if sampler == 'balanced':
        counts = df.Label.value_counts()
        weights = [1 / counts[label] for label in df.Label]
        sample = WeightedRandomSampler(weights, len(df), replacement=True, generator=generator)
    return DataLoader(DeepWeedsDataset(df, images_dir, transform), batch_size=batch_size,
                      shuffle=train and sample is None, sampler=sample,
                      drop_last=train and len(df) >= batch_size, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker,
                      generator=generator)
