"""Official DeepWeeds data and checkpoint restoration for Kaggle notebooks."""
import hashlib
import csv
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

MD5 = 'b7b30f96d466fba86016aa5a26606e0f'
LABEL_BASE = 'https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels'
IMAGE_URL = 'https://zenodo.org/records/7939060/files/images.zip?download=1'


def download(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + '.part')
    subprocess.run(['wget', '--tries=5', '--timeout=60', '-c', '-O', str(partial), url], check=True)
    partial.replace(destination)


def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    for member in archive.infolist():
        if not (destination / member.filename).resolve().is_relative_to(destination):
            raise ValueError('Unsafe archive path')
    archive.extractall(destination)


def prepare_data(root='/kaggle/temp/deepweeds-data', archive_path=None):
    """Fetch original fold-0 CSVs; verify images.zip, including an attached archive."""
    root = Path(root)
    labels = root / 'labels'
    labels.mkdir(parents=True, exist_ok=True)
    for name in ('labels', 'train_subset0', 'val_subset0', 'test_subset0'):
        destination = labels / f'{name}.csv'
        if not destination.exists():
            download(f'{LABEL_BASE}/{name}.csv', destination)
    # Read-only Kaggle input is never modified; extraction goes to temporary storage.
    attached = archive_path is not None
    archive_path = Path(archive_path) if attached else root / 'images.zip'
    if not archive_path.is_file():
        if attached:
            raise FileNotFoundError(archive_path)
        download(IMAGE_URL, archive_path)
    digest = hashlib.md5()
    with archive_path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    if digest.hexdigest() != MD5:
        if not attached:
            archive_path.unlink()
        raise RuntimeError('images.zip checksum failed; use the official archive or rerun to redownload')
    extracted = root / 'extracted'
    extracted.mkdir(exist_ok=True)
    marker = extracted / '.complete'
    if not marker.exists():
        with zipfile.ZipFile(archive_path) as archive:
            safe_extract(archive, extracted)
        marker.touch()
    import pandas as pd
    first = pd.read_csv(labels / 'train_subset0.csv').Filename.iloc[0]
    candidates = list(extracted.rglob(first))
    if len(candidates) != 1:
        raise RuntimeError(f'Cannot locate unique images directory for {first}')
    return candidates[0].parent, labels


def restore_results(source, output):
    """Copy a full Colab/Kaggle session directory or ZIP into writable storage."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError('RESUME_DIR must differ from output')
    marker = output / '.restored.json'
    if marker.exists():
        if json.loads(marker.read_text()) != {'source': str(source)}:
            raise ValueError('Output was restored from another source; use a new SESSION_NAME')
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Restore requires an empty output directory; use a new SESSION_NAME')
    def copy_session(directory):
        directory = Path(directory)
        candidates = ([directory] if (directory / 'runs').is_dir() else
                      [p.parent for p in directory.rglob('runs') if p.is_dir()])
        if len(candidates) != 1:
            raise ValueError('RESUME_DIR must contain exactly one session with runs/')
        shutil.copytree(candidates[0], output, dirs_exist_ok=True)
    if source.is_file() and zipfile.is_zipfile(source):
        with tempfile.TemporaryDirectory(prefix='deepweeds-resume-') as directory:
            with zipfile.ZipFile(source) as archive:
                safe_extract(archive, directory)
            copy_session(directory)
    elif source.is_dir():
        copy_session(source)
    else:
        raise FileNotFoundError(source)
    marker.write_text(json.dumps({'source': str(source)}), encoding='utf-8')


def migrate_paths(output, images_dir, labels_dir):
    """Rebase path fields only; preserve training settings, decisions and saved metrics."""
    output = Path(output).resolve()
    targets = {'images_dir': str(images_dir), 'labels_dir': str(labels_dir),
               'out_dir': str(output / 'runs'), 'pred_dir': str(output / 'predictions'),
               'curves_dir': str(output / 'curves')}
    audit_path = output / 'migration.json'
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    configs = sorted((output / 'runs').glob('*/seed*/config.json'))
    old_run_dirs = set(audit.get('old_run_dirs', []))
    old_run_dirs.update(json.loads(p.read_text())['out_dir'] for p in configs)
    metadata = list(configs)
    for name in ('summary.json', 'summary_ready.json'):
        metadata += sorted((output / 'runs').glob(f'*/seed*/{name}'))
    metadata += [p for p in (output / 'selection.json', output / 'inference_results.json') if p.exists()]

    def rewrite(value):
        if isinstance(value, list):
            return [rewrite(v) for v in value]
        if not isinstance(value, dict):
            return value
        result = {k: rewrite(v) for k, v in value.items()}
        if 'exp_id' in result and 'out_dir' in result:
            result.update(targets)
        if 'checkpoint' in result:
            checkpoint = Path(result['checkpoint'])
            for old_dir in sorted(old_run_dirs, key=len, reverse=True):
                try:
                    suffix = checkpoint.relative_to(old_dir)
                except ValueError:
                    continue
                result['checkpoint'] = str(output / 'runs' / suffix)
                break
            else:
                raise ValueError(f'Cannot migrate checkpoint path: {checkpoint}')
        return result

    changes = []
    for path in metadata:
        original = json.loads(path.read_text())
        updated = rewrite(original)
        if original != updated:
            changes.append((path, updated))
    if not changes:
        return []
    if not audit:
        audit['original_environments'] = {
            str(p.relative_to(output)): json.loads(p.read_text())
            for p in sorted((output / 'runs').glob('*/seed*/environment.json'))}
    audit.update({'old_run_dirs': sorted(old_run_dirs), 'target_paths': targets})
    # Keep an audit and immutable originals, including Colab hardware/version records.
    audit_path.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    for path, updated in changes:
        backup = output / 'migration_originals' / path.relative_to(output)
        if not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
        temporary = path.with_suffix('.migration.tmp')
        temporary.write_text(json.dumps(updated, indent=2), encoding='utf-8')
        temporary.replace(path)
    return [str(path.relative_to(output)) for path, _ in changes]


def resume_status(output):
    """Inspect saved histories and summaries without loading models or using test data."""
    output = Path(output)
    rows = []
    for path in sorted((output / 'runs').glob('*/seed*/config.json')):
        cfg = json.loads(path.read_text())
        directory = path.parent
        epoch = 0
        if (directory / 'history.csv').exists():
            with (directory / 'history.csv').open() as stream:
                history = list(csv.DictReader(stream))
            if history:
                epoch = int(history[-1]['epoch'])
        summary = (json.loads((directory / 'summary.json').read_text())
                   if (directory / 'summary.json').exists() else {})
        status = ('Test hoàn tất' if 'macro_f1_test' in summary else
                  'Train hoàn tất' if summary else 'Chưa hoàn tất')
        rows.append({'exp_id': cfg['exp_id'], 'seed': cfg['seed'], 'epoch': epoch,
                     'epochs': cfg['epochs'], 'status': status,
                     'latest.pt': (directory / 'latest.pt').exists(),
                     'gpu_measured': summary.get('latency', {}).get('gpu', '')})
    return rows
