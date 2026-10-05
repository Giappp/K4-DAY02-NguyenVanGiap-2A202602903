"""Fill the exact starter notebook with Kaggle cells and a reproducible source bundle."""
import base64
import copy
import io
import json
import textwrap
import zipfile
from pathlib import Path

SUBMISSION = Path(__file__).resolve().parent
ROOT = SUBMISSION.parents[1]
PAYLOAD_DECLARATION = 'SOURCE_B64 = "__PAYLOAD__"'

SETUP = '''
import base64, io, json, os, platform, shutil, subprocess, sys, zipfile
from pathlib import Path

if not Path("/kaggle/working").is_dir():
    raise RuntimeError("Import notebook này vào Kaggle; bật Accelerator GPU và Internet.")
# Mã nguồn đi kèm notebook, không cần upload ZIP hoặc clone GitHub.
SOURCE_B64 = "__PAYLOAD__"
REPO_DIR = Path("/kaggle/temp/deepweeds-lab")
REPO_DIR.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_B64))) as archive:
    for member in archive.infolist():
        if not (REPO_DIR / member.filename).resolve().is_relative_to(REPO_DIR.resolve()):
            raise ValueError("Đường dẫn bundle không hợp lệ")
    archive.extractall(REPO_DIR)
SUBMISSION_DIR = REPO_DIR / "submissions/2A202602903_NguyenVanGiap"
CODE_DIR = SUBMISSION_DIR / "code"
# Tạo lại notebook để code/ trong output chứa cả notebook chạy lại được.
subprocess.run([sys.executable, str(SUBMISSION_DIR / "build_kaggle_notebook.py")], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                str(SUBMISSION_DIR / "requirements-kaggle.txt")], check=True)
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(CODE_DIR))
import numpy as np, pandas as pd, torch, torchvision, timm
from IPython.display import display, FileLink
if not torch.cuda.is_available():
    raise RuntimeError("Chọn Settings → Accelerator → GPU trên Kaggle rồi chạy lại.")
print("Python:", platform.python_version(), "torch:", torch.__version__,
      "torchvision:", torchvision.__version__, "timm:", timm.__version__)
print("GPU:", torch.cuda.get_device_name(0), "VRAM (GB):",
      round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
print("Lab dùng GPU 0; các backbone dùng cùng batch và công thức.")
'''

CELLS = {
2: SETUP,
4: '''
# Full: 12 epoch, 3 seed chung kết. Smoke: 2 epoch, bỏ test.
MODE = "full"  # full | smoke
BATCH_SIZE = 32  # Nếu OOM, dùng 16 và SESSION_NAME mới cho mọi backbone.
SESSION_NAME = f"day2_{MODE}_batch{BATCH_SIZE}"
RESUME_DIR = None  # Thư mục session hoặc ZIP Colab/Kaggle trong /kaggle/input, có runs/.
IMAGES_ARCHIVE = None  # Tùy chọn: /kaggle/input/<dataset>/images.zip nguyên bản.
if MODE not in ("full", "smoke") or BATCH_SIZE < 1:
    raise ValueError("Kiểm tra MODE và BATCH_SIZE")
if Path(SESSION_NAME).name != SESSION_NAME:
    raise ValueError("SESSION_NAME phải là tên thư mục, không phải đường dẫn")
OUTPUT_DIR = Path("/kaggle/working") / SESSION_NAME
from kaggle_setup import prepare_data, restore_results, migrate_paths, resume_status
if RESUME_DIR:
    restore_results(RESUME_DIR, OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
shutil.copytree(CODE_DIR, OUTPUT_DIR / "code", dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__"))
shutil.copy2(REPO_DIR / "eval.py", OUTPUT_DIR / "eval.py")
shutil.copy2(SUBMISSION_DIR / "requirements-kaggle.txt", OUTPUT_DIR)
shutil.copy2(SUBMISSION_DIR / "README.md", OUTPUT_DIR)
if RESUME_DIR and (OUTPUT_DIR / "environment-pip.txt").exists():
    original_env = OUTPUT_DIR / "environment-pip-original.txt"
    if not original_env.exists():
        shutil.copy2(OUTPUT_DIR / "environment-pip.txt", original_env)
with (OUTPUT_DIR / "environment-pip.txt").open("w") as stream:
    subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=stream, check=True)
IMAGES_DIR, LABELS_DIR = prepare_data(archive_path=IMAGES_ARCHIVE)
if RESUME_DIR:
    changed = migrate_paths(OUTPUT_DIR, IMAGES_DIR, LABELS_DIR)
    print("Đã chuyển đường dẫn trong", len(changed), "file; giữ nguyên cấu hình/seed/kết quả cũ.")
    display(pd.DataFrame(resume_status(OUTPUT_DIR)))
# Lưu CSV gốc để chạy eval.py độc lập từ ZIP kết quả.
shutil.copytree(LABELS_DIR, OUTPUT_DIR / "labels", dirs_exist_ok=True)
print("Ảnh:", IMAGES_DIR, "Nhãn:", LABELS_DIR, "Output:", OUTPUT_DIR)
''',
5: '''
from train import Config, run
base = Config(images_dir=str(IMAGES_DIR), labels_dir=str(LABELS_DIR),
              out_dir=str(OUTPUT_DIR / "runs"), pred_dir=str(OUTPUT_DIR / "predictions"),
              curves_dir=str(OUTPUT_DIR / "curves"), batch_size=BATCH_SIZE,
              epochs=12 if MODE == "full" else 2)
stored_selection = (json.loads((OUTPUT_DIR / "selection.json").read_text())
                    if (OUTPUT_DIR / "selection.json").exists() else None)
print(base)
''',
7: '''
from experiments import (eda, pipeline_checks, compare_backbones, training_sweep,
                         inference_sweep, final_runs, export_results)
split_stats = eda(base)
display(pd.DataFrame(split_stats["per_class"]).sort_index())
print("Số ảnh:", split_stats["n"], "Giao:", split_stats["overlap"])
''',
8: '''
# Kiểm tra các kỹ thuật dễ sai; chạy hai bộ test ở process riêng để tách module.
subprocess.run([sys.executable, "-m", "pytest", "-q", str(SUBMISSION_DIR / "tests"),
                "-o", "testpaths="], cwd=REPO_DIR, check=True)
subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"],
                cwd=REPO_DIR, check=True)
pipeline_checks(base)  # seed, loss so với ln(9), overfit batch và ảnh augmentation.
''',
10: '''
backbone_results = compare_backbones(base)
backbone_table = pd.DataFrame([
    {k: v for k, v in r.items() if k not in ("config", "latency")} | r["latency"]
    for r in backbone_results])
display(backbone_table[["exp_id", "backbone", "weight_tag", "params_M", "GMAC",
                       "macro_f1_val", "top1_val", "train_seconds_per_epoch", "p95"]]
        .sort_values("macro_f1_val", ascending=False))
# Chọn theo F1 val; nếu hòa, ưu tiên p95 thấp hơn.
if stored_selection:
    backbone_winner = next(r for r in backbone_results
                           if r["exp_id"] == stored_selection["backbone_selection"])
else:
    backbone_winner = sorted(backbone_results,
                            key=lambda r: (-r["macro_f1_val"], r["latency"]["p95"]))[0]
print("Backbone đi tiếp:", backbone_winner["backbone"])
''',
12: '''
from dataclasses import replace
training_base = replace(base, backbone=backbone_winner["backbone"])
training_results = training_sweep(training_base)
display(pd.DataFrame(training_results)[["exp_id", "axis", "change", "macro_f1_val", "delta"]])
if stored_selection:
    training_winner = next(r for r in training_results
                           if r["exp_id"] == stored_selection["training"]["exp_id"])
    if training_winner["config"] != stored_selection["training"]:
        raise ValueError("Cấu hình khác quyết định đã chốt; kiểm tra dữ liệu khôi phục")
else:
    training_winner = sorted(training_results,
                            key=lambda r: (-r["macro_f1_val"], r["latency"]["p95"]))[0]
print("Công thức đi tiếp:", training_winner["exp_id"])
''',
14: '''
inference_results = inference_sweep(training_winner)
display(pd.DataFrame(inference_results)[["exp_id", "method", "K", "macro_f1_val",
                                        "ece_val", "p50", "p95", "p99", "dtype"]])
eligible = [r for r in inference_results
            if r["method"] in ("identity", "hflip_prob", "hflip_logit", "temperature")]
if stored_selection:
    inference_winner = stored_selection["inference"]
else:
    inference_winner = sorted(eligible,
                             key=lambda r: (-r["macro_f1_val"], r["ece_val"], r["p95"]))[0]
print("Suy luận chung kết:", inference_winner["method"])
# Lưu quyết định trên val trước khi xem test; không đổi khi tiếp tục phiên.
selection = {"training": training_winner["config"], "inference": inference_winner,
             "backbone_selection": backbone_winner["exp_id"]}
selection_path = OUTPUT_DIR / "selection.json"
if selection_path.exists():
    if json.loads(selection_path.read_text()) != selection:
        raise ValueError("Đã chốt cấu hình khác; dùng SESSION_NAME mới")
else:
    selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
''',
16: '''
final_results = []
if MODE == "full":
    final_results = final_runs(training_base, training_winner, inference_winner["method"])
    display(pd.DataFrame(final_results)[["exp_id", "seed", "macro_f1_val",
                                        "macro_f1_test", "top1_test", "ece_test"]])
else:
    print("Smoke mode: bỏ chung kết và test; kết quả chỉ để kiểm tra luồng.")
''',
17: '''
# eval.py gốc: tính lại từ CSV đã lưu, không forward test lần nữa.
if final_results:
    common = ["--test-csv", str(LABELS_DIR / "test_subset0.csv"),
              "--labels", str(LABELS_DIR / "labels.csv"),
              "--out", str(OUTPUT_DIR / "eval_out")]
    for tag in ("F01", "T00"):
        subprocess.run([sys.executable, str(REPO_DIR / "eval.py"), "score", "--pred",
                        str(OUTPUT_DIR / f"predictions/{tag}_seed*_test.csv"),
                        "--tag", tag, *common], check=True)
else:
    print("Chưa có kết quả test để score.")
''',
18: '''
if final_results:
    grade = [sys.executable, str(REPO_DIR / "eval.py"), "grade",
             "--final", str(OUTPUT_DIR / "predictions/F01_seed*_test.csv"),
             "--baseline", str(OUTPUT_DIR / "predictions/T00_seed*_test.csv"),
             "--final-val", str(OUTPUT_DIR / "predictions/F01_seed*_val.csv"),
             "--val-csv", str(LABELS_DIR / "val_subset0.csv"),
             "--latency-p95-ms", str(max(r["latency"]["p95"] for r in final_results
                                        if r["exp_id"] == "F01")), *common]
    if inference_winner["method"] == "temperature":
        grade += ["--uncal", str(OUTPUT_DIR / "predictions/F01uncal_seed*_test.csv")]
    subprocess.run(grade, check=True)
else:
    print("Smoke mode: không tự chấm phần test.")
''',
20: '''
xlsx = export_results(training_base, backbone_results, training_results,
                      inference_results, final_results)
archive_path = Path("/kaggle/working") / f"{SESSION_NAME}_results.zip"
with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
    for folder in ("code", "curves", "predictions", "eval_out", "labels", "migration_originals"):
        for path in sorted((OUTPUT_DIR / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, path.relative_to(OUTPUT_DIR))
    for name in ("results.xlsx", "report.md", "all_results.json", "selection.json",
                 "requirements-kaggle.txt", "environment-pip.txt", "environment-pip-original.txt",
                 "migration.json", "README.md", "eval.py"):
        path = OUTPUT_DIR / name
        if path.exists():
            archive.write(path, name)
    for path in sorted((OUTPUT_DIR / "runs").rglob("*")):
        if path.is_file() and path.suffix in (".json", ".csv"):
            archive.write(path, path.relative_to(OUTPUT_DIR))
print("Excel:", xlsx, "Báo cáo:", OUTPUT_DIR / "report.md")
print("ZIP:", archive_path, "— checkpoint .pt giữ trong runs/, không đưa vào ZIP nộp bài.")
# Đường dẫn tương đối để tải qua giao diện Notebook.
os.chdir("/kaggle/working")
display(FileLink(archive_path.name))
print("Chọn Save Version → Save & Run All để lưu output của phiên trên Kaggle.")
''',
}

NOTES = {
3: 'Trên Kaggle, dữ liệu được giải nén vào `/kaggle/temp` và output lưu ở '
   '`/kaggle/working/<SESSION_NAME>`. Có thể gắn `images.zip` nguyên bản qua Input; '
   'CSV fold 0 vẫn tải từ tác giả. Archive được kiểm MD5 trước khi dùng.\n\n'
   '`MODE="full"`: 12 epoch, batch 32 chung cho mọi backbone; batch được giảm từ 64 '
   'của công thức tham khảo để vừa VRAM. `MODE="smoke"`: 2 epoch, không đánh giá test. '
   '`RESUME_DIR` trỏ đến thư mục hoặc ZIP phiên Colab/Kaggle trước có `runs/`; '
   'tải cả session gồm predictions, curves, selection.json và inference_results.json. '
   'Đường dẫn cũ được chuyển tự động; giữ nguyên cấu hình và tên phiên. '
   'Bảng trạng thái sau setup cho biết seed nào cần chạy tiếp. Chạy Bước 1–3 để nạp '
   'kết quả/quyết định cũ, sau đó Bước 4 tiếp tục chung kết và Bước 5 xuất sản phẩm.',
11: 'Đã thực hiện ba trục: finetune/frozen, basic/color, CE/label smoothing/focal. '
    '`T05` kết hợp các yếu tố cải thiện validation. Vòng này dùng một seed; '
    'độ ổn định được đánh giá ở chung kết.',
13: 'Đã thực hiện I00 một view và 5 phương pháp: TTA flip gộp probability/logit, '
    'temperature scaling, gộp BN, AMP. Đo batch 1, 10 warmup, 100 lần và đồng bộ CUDA. '
    'Chung kết chọn trong identity/TTA/temperature theo macro-F1, ECE rồi p95 trên val.',
15: 'Các seed là 0/1/2. Mốc dùng backbone đã chọn và công thức T00. '
    'Run đã hoàn tất được đọc lại; CSV/logit test đã lưu được tái sử dụng khi khôi phục. '
    'Nếu phiên dừng trước khi ghi dự đoán thành công, phần chưa lưu phải chạy tiếp.',
19: 'Ô cuối tạo đủ 7 sheet và ZIP tải xuống. Báo cáo lấy số liệu thật; '
    'xem ảnh lỗi để bổ sung nhận xét. Smoke output chỉ kiểm tra luồng, không dùng nộp bài. '
    'Checkpoint giữ trong output để tiếp tục phiên; ZIP bài nộp loại checkpoint và ảnh dữ liệu.',
}


def source_bundle():
    files = [ROOT / name for name in ('eval.py', 'README.md', 'GUIDE.md', 'RUBRIC.md')]
    files += sorted((ROOT / 'starter').glob('*.py'))
    files += [ROOT / 'starter/lab_day2.ipynb']
    files += sorted((ROOT / 'tests').glob('*.py'))
    files += [p for p in sorted((SUBMISSION / 'code').glob('*.py')) if p.name != 'colab_setup.py']
    files += sorted((SUBMISSION / 'tests').glob('*.py'))
    files += [SUBMISSION / name for name in ('README.md', 'HUONG_DAN_CODE.md',
                                           'requirements-kaggle.txt', 'build_kaggle_notebook.py')]
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    return base64.b64encode(stream.getvalue()).decode('ascii')


def build_notebook():
    notebook = copy.deepcopy(json.loads((ROOT / 'starter/lab_day2.ipynb').read_text()))
    notebook['cells'][0]['source'] = [
        '# Lab Day 2 — Backbone, công thức huấn luyện và suy luận trên DeepWeeds\n',
        '\n',
        'Bài làm **Nguyễn Văn Giáp · 2A202602903**, hoàn thiện theo đúng 21 ô của `starter/lab_day2.ipynb`.\n',
        '\n',
        '**Kaggle:** Import Notebook → chọn file này → Settings: GPU và Internet On. '
        'Chạy từ trên xuống hoặc Save & Run All. Mã nguồn đã đóng gói trong ô setup, '
        'không cần Drive, ZIP riêng hay GitHub. Giữ `MODE="full"` để làm đầy đủ.\n',
        '\n',
        'Đọc `README.md`, `GUIDE.md` và `RUBRIC.md` của đề bài. '
        'Giữ split fold 0, chọn cấu hình chỉ trên val và dùng `eval.py` gốc.\n',
    ]
    payload = source_bundle()
    for index, source in CELLS.items():
        source = textwrap.dedent(source).strip() + '\n'
        if index == 2:
            source = source.replace(PAYLOAD_DECLARATION, f'SOURCE_B64 = "{payload}"')
        notebook['cells'][index]['source'] = source.splitlines(keepends=True)
    for index, note in NOTES.items():
        original = ''.join(notebook['cells'][index]['source'])
        notebook['cells'][index]['source'] = (original.rstrip() + '\n\n' + note + '\n').splitlines(keepends=True)
    for index, cell in enumerate(notebook['cells']):
        cell['id'] = f'day2-{index:02d}'
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
    notebook['metadata'] = {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11'},
        'kaggle': {'isGpuEnabled': True, 'isInternetEnabled': True},
    }
    notebook['nbformat'] = 4
    notebook['nbformat_minor'] = 5
    encoded = json.dumps(notebook, ensure_ascii=False, indent=1) + '\n'
    for path in (SUBMISSION / 'lab_day2.ipynb', SUBMISSION / 'code/lab_day2.ipynb'):
        path.write_text(encoded, encoding='utf-8')
    print('Created Kaggle notebook (21 starter cells, embedded source):', SUBMISSION / 'lab_day2.ipynb')
    return notebook


if __name__ == '__main__':
    build_notebook()
