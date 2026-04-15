from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(__file__).resolve().parent / "isothermal_dataset"
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
TRAIN_SCRIPT = PROJECT_ROOT / "src" / "models" / "PGSSI" / "PGSSI_train.py"


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate PGSSI on isothermal splits.")
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_ROOT))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--init-model-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def temperature_dirs(root: Path) -> list[Path]:
    return sorted([path for path in root.iterdir() if path.is_dir()], key=lambda p: float(p.name))


def prepare_test_file(temp_dir: Path, prepared_dir: Path) -> Path:
    temp_name = temp_dir.name
    raw_test = pd.read_csv(temp_dir / f"{temp_name}_test.csv")
    test_df = raw_test.copy()
    if "T" not in test_df.columns:
        if "T_K" not in test_df.columns:
            raise KeyError(f"{temp_name}_test.csv has neither T nor T_K")
        test_df["T"] = test_df["T_K"].astype(float) - 273.15
    prepared_path = prepared_dir / f"{temp_name}_test_prepared.csv"
    test_df.to_csv(prepared_path, index=False)
    return prepared_path


def run_one_temperature(temp_dir: Path, args, output_dir: Path, prepared_dir: Path) -> dict:
    temp_name = temp_dir.name
    run_dir = output_dir / temp_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_path = temp_dir / f"{temp_name}_train.csv"
    valid_path = temp_dir / f"{temp_name}.csv"
    test_path = prepare_test_file(temp_dir, prepared_dir)

    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--run-dir",
        str(run_dir),
        "--train-path",
        str(train_path),
        "--valid-path",
        str(valid_path),
        "--test-path",
        str(test_path),
        "--model-name",
        "PGSSI_Isothermal",
        "--n-epochs",
        str(args.epochs),
        "--hidden-dim",
        str(args.hidden_dim),
        "--lr",
        str(args.lr),
        "--batch-size",
        str(args.batch_size),
        "--checkpoint-interval",
        str(args.checkpoint_interval),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--quiet-progress",
        "--seed",
        str(args.seed),
    ]
    if args.init_model_path:
        cmd.extend(["--init-model-path", str(args.init_model_path)])

    print(f"\n=== Temperature {temp_name} °C ===")
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

    artifact_prefix = f"{temp_name}_train_PGSSI_Isothermal"
    metrics_path = run_dir / f"{artifact_prefix}_{test_path.stem}_metrics.json"
    training_path = run_dir / f"{artifact_prefix}_training.csv"

    with open(metrics_path, "r", encoding="utf-8") as fh:
        test_metrics = json.load(fh)
    training_df = pd.read_csv(training_path)

    best_valid_idx = training_df["MAE_Valid"].idxmin()
    return {
        "temperature_C": float(temp_name),
        "epochs_ran": int(len(training_df)),
        "best_valid_mae": float(training_df.loc[best_valid_idx, "MAE_Valid"]),
        "best_valid_r2": float(training_df.loc[best_valid_idx, "R2_Valid"]),
        "test_mae": float(test_metrics.get("mae", float("nan"))),
        "test_r2": float(test_metrics.get("r2", float("nan"))),
        "test_ae_le_01_pct": float(test_metrics.get("AE<=0.1", float("nan"))),
        "test_ae_le_02_pct": float(test_metrics.get("AE<=0.2", float("nan"))),
        "test_ae_le_03_pct": float(test_metrics.get("AE<=0.3", float("nan"))),
        "num_samples_total": int(test_metrics.get("num_samples_total", 0)),
        "run_dir": str(run_dir),
    }


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    prepared_dir = output_dir / "prepared_test_sets"
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for temp_dir in temperature_dirs(data_root):
        rows.append(run_one_temperature(temp_dir, args, output_dir, prepared_dir))

    result_df = pd.DataFrame(rows).sort_values("temperature_C")
    csv_path = output_dir / "isothermal_results.csv"
    md_path = output_dir / "isothermal_results.md"
    result_df.to_csv(csv_path, index=False)
    result_df.to_markdown(md_path, index=False)

    print(f"\nSaved isothermal results to: {csv_path}")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
