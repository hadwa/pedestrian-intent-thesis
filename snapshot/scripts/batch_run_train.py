import sys
import subprocess
import argparse
from pathlib import Path

# ...existing code...
REPO_ROOT = Path(__file__).resolve().parents[1]

# list entries can be either a single csv file or a directory containing CSVs
CSV_DIRS = [
    ("src/train_10_enriched.csv", "results/v2"),
    ("src/train_10.csv", "results/v2"),
    ("src/train_11_kp.csv", "results/v2"),
    ("src/train_11.csv", "results/v2"),
    ("src/train_15.csv", "results/v2"),
]

PY = sys.executable
TRAIN_SCRIPT_DEFAULT = REPO_ROOT / "src" / "train_tabular_factory_v2.py"

def run_one(csv_path: Path, outdir: Path, model: str, train_script: Path, extra_args=None):
    outdir.mkdir(parents=True, exist_ok=True)
    log = outdir / "train.log"
    cmd = [PY, str(train_script), "--csv", str(csv_path), "--outdir", str(outdir), "--model", model]
    if extra_args:
        cmd += extra_args
    print("Running:", " ".join(cmd))
    with open(log, "wb") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
        proc.wait()
        return proc.returncode

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", "-m", nargs="+", default=["rf"],
                    help="Models to run per CSV (e.g. rf xgb lgbm cat)")
    ap.add_argument("--train-script", type=str, default=str(TRAIN_SCRIPT_DEFAULT),
                    help="Path to train_tabular_factory_v2.py")
    ap.add_argument("--extra-args", "-e", nargs="*", default=[],
                    help="Extra args forwarded to the training script")
    args = ap.parse_args()

    train_script = Path(args.train_script)
    if not train_script.exists():
        print("Train script not found:", train_script)
        return

    for src_str, dst_root_str in CSV_DIRS:
        # Resolve relative paths against repo root
        src = Path(src_str)
        dst = Path(dst_root_str)
        if not src.is_absolute():
            src = REPO_ROOT / src
        if not dst.is_absolute():
            dst = REPO_ROOT / dst

        if not src.exists():
            print("skip, not found:", src)
            continue

        if src.is_file():
            csvs = [src]
        else:
            csvs = sorted(list(src.glob("*.csv")))
            if not csvs:
                print("skip, no CSVs in:", src)
                continue

        for csv in csvs:
            for model in args.models:
            
                outdir = dst / csv.stem / model
                rc = run_one(csv, outdir, model, train_script, extra_args=args.extra_args)
                print(f"Finished {csv} -> {outdir} rc={rc}")

if __name__ == "__main__":
    main()