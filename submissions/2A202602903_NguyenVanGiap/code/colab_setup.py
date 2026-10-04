"""Download official DeepWeeds data on Colab with verified archive and atomic downloads."""
import hashlib
import shutil
import subprocess
import zipfile
from pathlib import Path

MD5 = 'b7b30f96d466fba86016aa5a26606e0f'


def download(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + '.part')
    subprocess.run(['wget', '--tries=5', '--timeout=60', '-c', '-O', str(part), url], check=True)
    part.replace(destination)


def prepare_data(root='/content/deepweeds-data', cache_dir=None):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    labels = root / 'labels'; labels.mkdir(exist_ok=True)
    base = 'https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels'
    for name in ('labels', 'train_subset0', 'val_subset0', 'test_subset0'):
        # Always fetch original CSVs; do not use modified local copies.
        download(f'{base}/{name}.csv', labels / f'{name}.csv')
    archive = root / 'images.zip'
    cache = Path(cache_dir) / 'images.zip' if cache_dir else None
    if not archive.exists():
        if cache and cache.exists():
            shutil.copy2(cache, archive)
        else:
            download('https://zenodo.org/records/7939060/files/images.zip?download=1', archive)
    digest = hashlib.md5()
    with archive.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    if digest.hexdigest() != MD5:
        archive.unlink()
        if cache and cache.exists():
            cache.unlink()
        raise RuntimeError('images.zip checksum failed; rerun this cell to redownload')
    if cache and not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        partial = cache.with_suffix('.part'); shutil.copy2(archive, partial); partial.replace(cache)
    extracted = root / 'extracted'; extracted.mkdir(exist_ok=True)
    marker = extracted / '.complete'
    if not marker.exists():
        with zipfile.ZipFile(archive) as z:
            for entry in z.infolist():
                target = (extracted / entry.filename).resolve()
                if not target.is_relative_to(extracted.resolve()):
                    raise ValueError('Unsafe archive path')
            z.extractall(extracted)
        marker.touch()
    import pandas as pd
    first = pd.read_csv(labels / 'train_subset0.csv').Filename.iloc[0]
    candidates = list(extracted.rglob(first))
    if len(candidates) != 1:
        raise RuntimeError(f'Cannot locate unique images directory for {first}')
    return candidates[0].parent, labels
