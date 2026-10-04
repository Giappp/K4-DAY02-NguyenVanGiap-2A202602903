"""Package source files only; no imports/training, dataset, checkpoints or local env."""
from pathlib import Path
import zipfile


def build_bundle():
    submission = Path(__file__).resolve().parent
    root = submission.parents[1]
    sources = [root / name for name in ('eval.py', 'README.md', 'GUIDE.md', 'RUBRIC.md')]
    for folder, extensions in [(root / 'starter', {'.py', '.ipynb'}),
                               (root / 'tests', {'.py'}),
                               (submission / 'code', {'.py', '.ipynb'}),
                               (submission / 'tests', {'.py'})]:
        sources.extend(p for p in folder.iterdir() if p.is_file() and p.suffix in extensions)
    sources.extend(submission / name for name in ('README.md', 'requirements-colab.txt', 'build_colab_bundle.py'))
    output = root / 'colab_bundle.zip'
    temporary = output.with_suffix('.tmp.zip')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(sources):
            archive.write(path, path.relative_to(root))
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
    temporary.replace(output)
    print(f'Created {output} ({output.stat().st_size:,} bytes, {len(sources)} files)')
    return output


if __name__ == '__main__':
    build_bundle()
