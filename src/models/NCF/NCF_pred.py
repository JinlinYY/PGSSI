from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

try:  # pragma: no cover
    from .data_utils import NCFDataset, encode_components, load_dataset
    from .model import NCFConfig, NCFModel
except ImportError:  # pragma: no cover
    current_dir = Path(__file__).resolve().parent
    sys.path.append(str(current_dir))
    from data_utils import NCFDataset, encode_components, load_dataset  # type: ignore  # noqa: E402
    from model import NCFConfig, NCFModel  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use trained NCF models to predict ln γ∞.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("dataset/public dataset"),
        help="Path to dataset CSV, or a directory containing CSV files.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("src/models/NCF/outputs"),
        help="Directory containing checkpoints.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="best.pt",
        help="Glob pattern for checkpoints within the directory (single model default: best.pt).",
    )
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for inference.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/models/NCF/outputs/predictions.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def load_checkpoints(checkpoint_dir: Path, pattern: str) -> List[Path]:
    files = sorted(checkpoint_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No checkpoints matching pattern '{pattern}' found in {checkpoint_dir}")
    return files


def predict_with_checkpoint(
    checkpoint_path: Path,
    frame,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_dict = checkpoint["config"]
    config = NCFConfig(
        num_solutes=config_dict["num_solutes"],
        num_solvents=config_dict["num_solvents"],
        embedding_dim=config_dict["embedding_dim"],
        representation_hidden_sizes=config_dict["representation_hidden_sizes"],
        collaborative_hidden_sizes=config_dict["collaborative_hidden_sizes"],
        dropout=config_dict["dropout"],
    )
    model = NCFModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = NCFDataset(
        frame=frame,
        temp_mean=checkpoint["temp_mean"],
        temp_std=checkpoint["temp_std"],
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    predictions: List[np.ndarray] = []
    with torch.no_grad():
        for solute_ids, solvent_ids, temperatures, _ in dataloader:
            solute_ids = solute_ids.to(device)
            solvent_ids = solvent_ids.to(device)
            temperatures = temperatures.to(device)
            preds = model(solute_ids, solvent_ids, temperatures).cpu().numpy().reshape(-1)
            predictions.append(preds)
    return np.concatenate(predictions, axis=0)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    raw_df = load_dataset(args.data)
    encoded = encode_components(raw_df)

    checkpoints = load_checkpoints(args.checkpoint_dir, args.pattern)

    predictions = []
    for ckpt in checkpoints:
        print(f"Generating predictions with {ckpt}...")
        preds = predict_with_checkpoint(ckpt, encoded.frame, args.batch_size, device)
        predictions.append(preds)

    ensemble_pred = np.mean(np.vstack(predictions), axis=0)
    output_df = encoded.frame.copy()
    output_df["predicted_log_gamma"] = ensemble_pred
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)
    print(f"Stored predictions in {args.output}")


if __name__ == "__main__":
    main()









