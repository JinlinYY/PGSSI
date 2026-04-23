'''默认会从最新的检查点开始训练'''
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

try:  # pragma: no cover
    from .data_utils import (  # type: ignore
        NCFDataset,
        build_component_vocabs,
        compute_temperature_stats,
        encode_frame,
        load_dataset,
    )
    from .model import NCFConfig, NCFModel
except ImportError:  # pragma: no cover
    current_dir = Path(__file__).resolve().parent
    sys.path.append(str(current_dir))
    from data_utils import (  # type: ignore  # noqa: E402
        NCFDataset,
        build_component_vocabs,
        compute_temperature_stats,
        encode_frame,
        load_dataset,
    )
    from model import NCFConfig, NCFModel  # type: ignore  # noqa: E402

DEFAULT_SEED = 42
MODEL_NAME = "NCF"
BEST_HYPERPARAMS: Dict[str, Any] = {
    "embedding_dim": 128,
    "repr_layer_sizes": [128, 128],
    "cf_layer_sizes": [256, 256, 256],
    "dropout": 0.0,
    "activation": "relu",
    "learning_rate": 0.001,
    "batch_size": 64,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train neural collaborative filtering model for γ∞ prediction.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("dataset/all"),
        help="Dataset directory (expects all_merged_train.csv / all_merged_valid.csv / all_merged_test.csv).",
    )
    parser.add_argument("--epochs", type=int, default=150, help="Maximum number of epochs.")
    parser.add_argument("--batch-size", type=int, default=BEST_HYPERPARAMS["batch_size"], help="Training batch size.")
    parser.add_argument("--lr", type=float, default=BEST_HYPERPARAMS["learning_rate"], help="Learning rate for Adam optimizer.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for Adam optimizer.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (in epochs).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/models/NCF/outputs"),
        help="Directory to store checkpoints.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of dataloader workers.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device.")
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable auto-resume and start training from scratch.",
    )
    parser.set_defaults(resume=True)
    return parser.parse_args()


def set_random_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def create_dataloaders(
    train_frame,
    val_frame,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, Tuple[float, float]]:
    temp_mean, temp_std = compute_temperature_stats(train_frame)
    train_dataset = NCFDataset(train_frame, temp_mean, temp_std)
    val_dataset = NCFDataset(val_frame, temp_mean, temp_std)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, (temp_mean, temp_std)


def train_fold(
    fold_idx: int,
    model: NCFModel,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    patience: int,
    start_epoch: int = 1,
    output_dir: Optional[Path] = None,
    config: Optional[NCFConfig] = None,
    temp_mean: Optional[float] = None,
    temp_std: Optional[float] = None,
    training_params: Optional[Dict[str, Any]] = None,
    checkpoint_every: int = 20,
    test_loader: Optional[DataLoader] = None,
    report_fn=None,
) -> Dict[str, float]:
    criterion = nn.MSELoss()
    best_val_mae = float("inf")
    epochs_without_improvement = 0
    best_state_dict = None

    if start_epoch < 1:
        start_epoch = 1

    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs + 1):
        last_epoch = epoch
        train_metrics = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, criterion, device, train=False)

        improved = val_metrics["mae"] < best_val_mae - 1e-6
        if improved:
            best_val_mae = val_metrics["mae"]
            epochs_without_improvement = 0
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        msg = (
            f"[Fold {fold_idx:02d}] Epoch {epoch:03d} | "
            f"Train MAE: {train_metrics['mae']:.4f} | Train R2: {train_metrics['r2']:.4f} | "
            f"Val MAE: {val_metrics['mae']:.4f} | Val R2: {val_metrics['r2']:.4f}"
        )
        if report_fn is not None:
            report_fn(msg)
        else:
            print(msg)

        if (
            output_dir is not None
            and config is not None
            and temp_mean is not None
            and temp_std is not None
            and checkpoint_every > 0
            and epoch % checkpoint_every == 0
        ):
            save_checkpoint(
                checkpoint_path=output_dir / f"epoch_{epoch:03d}.pt",
                model=model,
                config=config,
                temp_mean=float(temp_mean),
                temp_std=float(temp_std),
                metrics={
                    "train_mse": float(train_metrics["mse"]),
                    "train_mae": float(train_metrics["mae"]),
                    "train_r2": float(train_metrics["r2"]),
                    "val_mse": float(val_metrics["mse"]),
                    "val_mae": float(val_metrics["mae"]),
                    "val_r2": float(val_metrics["r2"]),
                    "best_val_mae": float(best_val_mae),
                },
                duration_seconds=0.0,
                training_params=training_params or {},
                epoch=epoch,
                optimizer_state_dict=optimizer.state_dict(),
            )
            if test_loader is not None:
                test_metrics = predict_metrics(model, test_loader, device)
                msg = (
                    f"[Fold {fold_idx:02d}] Epoch {epoch:03d} | "
                    f"Test MAE: {test_metrics['mae']:.4f} | Test R2: {test_metrics['r2']:.4f}"
                )
                if report_fn is not None:
                    report_fn(msg)
                else:
                    print(msg)

        if epochs_without_improvement >= patience:
            print(f"[Fold {fold_idx:02d}] Early stopping triggered after {epoch} epochs.")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    final_train_metrics = run_epoch(model, train_loader, optimizer, criterion, device, train=False)
    final_val_metrics = run_epoch(model, val_loader, optimizer, criterion, device, train=False)
    return {
        "train_mse": final_train_metrics["mse"],
        "train_mae": final_train_metrics["mae"],
        "train_r2": final_train_metrics["r2"],
        "val_mse": final_val_metrics["mse"],
        "val_mae": final_val_metrics["mae"],
        "val_r2": final_val_metrics["r2"],
        "last_epoch": float(last_epoch),
    }


def run_epoch(
    model: NCFModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    train: bool,
) -> Dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total_abs_error = 0.0
    residual_sum_squares = 0.0
    target_sum = 0.0
    target_sum_squares = 0.0
    total_samples = 0

    for solute_ids, solvent_ids, temperatures, targets in dataloader:
        solute_ids = solute_ids.to(device)
        solvent_ids = solvent_ids.to(device)
        temperatures = temperatures.to(device)
        targets = targets.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            predictions = model(solute_ids, solvent_ids, temperatures)
            loss = criterion(predictions, targets)
            if train:
                loss.backward()
                optimizer.step()

        batch_size = targets.shape[0]
        total_loss += loss.item() * batch_size
        differences = predictions - targets
        total_abs_error += torch.sum(torch.abs(differences)).item()
        residual_sum_squares += torch.sum(differences.pow(2)).item()
        target_sum += torch.sum(targets).item()
        target_sum_squares += torch.sum(targets.pow(2)).item()
        total_samples += batch_size

    mse = total_loss / total_samples
    mae = total_abs_error / total_samples
    target_mean = target_sum / total_samples
    ss_tot = target_sum_squares - total_samples * target_mean * target_mean
    if ss_tot <= 1e-12:
        r2 = 0.0
    else:
        r2 = 1.0 - (residual_sum_squares / ss_tot)
    return {"mse": mse, "mae": mae, "r2": r2}


def predict_metrics(model: NCFModel, dataloader: DataLoader, device: torch.device):
    """Return metrics dict for evaluation."""
    model.eval()
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    with torch.no_grad():
        for solute_ids, solvent_ids, temperatures, targets in dataloader:
            solute_ids = solute_ids.to(device)
            solvent_ids = solvent_ids.to(device)
            temperatures = temperatures.to(device)
            targets = targets.to(device)
            preds = model(solute_ids, solvent_ids, temperatures)
            ys.append(targets.detach().cpu().numpy().reshape(-1))
            ps.append(preds.detach().cpu().numpy().reshape(-1))

    y_true = np.concatenate(ys) if ys else np.array([], dtype=np.float64)
    y_pred = np.concatenate(ps) if ps else np.array([], dtype=np.float64)
    if y_true.size == 0:
        return {"mse": 0.0, "mae": 0.0, "r2": 0.0}
    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff**2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    r2 = 0.0 if ss_tot <= 1e-12 else float(1.0 - (ss_res / ss_tot))
    return {"mse": mse, "mae": mae, "r2": r2}


def load_public_splits(path: Path):
    path = Path(path)
    if path.is_dir():
        # Prefer unified "all_merged_*.csv" splits (new default).
        train_path = path / "all_merged_train.csv"
        valid_path = path / "all_merged_valid.csv"
        test_path = path / "all_merged_test.csv"
        if not train_path.exists() or not valid_path.exists():
            # Backward compat with the original public dataset naming.
            train_path = path / "molecule_train.csv"
            valid_path = path / "molecule_valid.csv"
            test_path = path / "molecule_test.csv"
        train_df = load_dataset(train_path)
        valid_df = load_dataset(valid_path)
        test_df = load_dataset(test_path) if test_path.exists() else None
        return train_df, valid_df, test_df
    df = load_dataset(path)
    # If a single file is provided, fall back to training with an internal split not provided here.
    # The user asked to use the existing public split, so we require directory mode.
    raise ValueError(
        "Expected a directory containing molecule_train.csv / molecule_valid.csv / molecule_test.csv. "
        f"Got file: {path}"
    )


def save_checkpoint(
    checkpoint_path: Path,
    model: NCFModel,
    config: NCFConfig,
    temp_mean: float,
    temp_std: float,
    metrics: Dict[str, float],
    duration_seconds: float,
    training_params: Dict[str, Any],
    epoch: Optional[int] = None,
    optimizer_state_dict: Optional[Dict[str, Any]] = None,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer_state_dict,
            "epoch": epoch,
            "config": asdict(config),
            "temp_mean": temp_mean,
            "temp_std": temp_std,
            "metrics": metrics,
            "duration_seconds": duration_seconds,
            "training_params": training_params,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")


def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    if not output_dir.exists():
        return None

    best_path = output_dir / "best.pt"
    epoch_ckpts = list(output_dir.glob("epoch_*.pt"))
    best_epoch = -1
    best_epoch_path: Optional[Path] = None
    for ckpt in epoch_ckpts:
        name = ckpt.stem  # e.g. epoch_020
        try:
            epoch_str = name.split("_", 1)[1]
            epoch_val = int(epoch_str)
        except (IndexError, ValueError):
            continue
        if epoch_val > best_epoch:
            best_epoch = epoch_val
            best_epoch_path = ckpt

    if best_epoch_path is not None:
        return best_epoch_path
    if best_path.exists():
        return best_path
    return None


def main() -> None:
    args = parse_args()
    set_random_seeds(args.seed)
    device = torch.device(args.device)
    print(f"device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    train_df, valid_df, test_df = load_public_splits(args.data)
    all_frames = [train_df, valid_df]
    if test_df is not None:
        all_frames.append(test_df)
    all_df = pd.concat(all_frames, ignore_index=True)

    solute_vocab, solvent_vocab = build_component_vocabs(all_df)
    train_frame = encode_frame(train_df, solute_vocab, solvent_vocab).reset_index(drop=True)
    valid_frame = encode_frame(valid_df, solute_vocab, solvent_vocab).reset_index(drop=True)
    test_frame = encode_frame(test_df, solute_vocab, solvent_vocab).reset_index(drop=True) if test_df is not None else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"Report_training_{MODEL_NAME}.txt"
    report_fp = report_path.open("w", encoding="utf-8")

    def print_report(msg: str) -> None:
        print(msg)
        report_fp.write(msg + "\n")
        report_fp.flush()

    config = NCFConfig(
        num_solutes=len(solute_vocab),
        num_solvents=len(solvent_vocab),
        embedding_dim=BEST_HYPERPARAMS["embedding_dim"],
        representation_hidden_sizes=BEST_HYPERPARAMS["repr_layer_sizes"],
        collaborative_hidden_sizes=BEST_HYPERPARAMS["cf_layer_sizes"],
        dropout=BEST_HYPERPARAMS["dropout"],
        activation=BEST_HYPERPARAMS["activation"],
    )
    batch_size = args.batch_size
    lr = args.lr

    print(
        "Using fixed hyperparameters: "
        f"embedding_dim={config.embedding_dim}, "
        f"repr_layers={config.representation_hidden_sizes}, "
        f"cf_layers={config.collaborative_hidden_sizes}, "
        f"dropout={config.dropout}, activation={config.activation}. "
        f"Batch size={batch_size}, learning rate={lr}."
    )

    run_start = time.time()
    train_loader, val_loader, (temp_mean, temp_std) = create_dataloaders(
        train_frame, valid_frame, batch_size, args.num_workers
    )

    checkpoint_path = args.output_dir / "best.pt"
    model = NCFModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    test_loader: Optional[DataLoader] = None
    if test_frame is not None:
        test_loader = DataLoader(
            NCFDataset(test_frame, temp_mean, temp_std),
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    start_epoch = 1
    if args.resume:
        latest_checkpoint = find_latest_checkpoint(args.output_dir)
        if latest_checkpoint is not None:
            print(f"Resuming from checkpoint: {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            if checkpoint.get("optimizer_state_dict") is not None:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            loaded_epoch = checkpoint.get("epoch")
            if isinstance(loaded_epoch, int) and loaded_epoch >= 1:
                start_epoch = loaded_epoch + 1

    metrics = train_fold(
        1,
        model,
        optimizer,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        patience=args.patience,
        start_epoch=start_epoch,
        output_dir=args.output_dir,
        config=config,
        temp_mean=temp_mean,
        temp_std=temp_std,
        training_params={
            "learning_rate": lr,
            "batch_size": batch_size,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "patience": args.patience,
        },
        checkpoint_every=30,
        test_loader=test_loader,
        report_fn=print_report,
    )

    test_metrics: Optional[Dict[str, float]] = None
    if test_loader is not None:
        test_metrics = predict_metrics(model, test_loader, device)
        test_json_path = args.output_dir / "test_results_best.json"
        with test_json_path.open("w", encoding="utf-8") as fp:
            json.dump(
                {
                    "model": MODEL_NAME,
                    "mae": float(test_metrics["mae"]),
                    "r2": float(test_metrics["r2"]),
                    "mse": float(test_metrics["mse"]),
                    "n_samples": int(test_frame.shape[0]) if test_frame is not None else 0,
                },
                fp,
                indent=2,
            )
        print_report(f"[{MODEL_NAME}] Final Test | MAE: {test_metrics['mae']:.6f} | R2: {test_metrics['r2']:.6f} | Saved: {test_json_path}")

    duration = time.time() - run_start
    metrics_out: Dict[str, Any] = {
        "train": {k.replace("train_", ""): v for k, v in metrics.items() if k.startswith("train_")},
        "valid": {k.replace("val_", ""): v for k, v in metrics.items() if k.startswith("val_")},
        "duration_seconds": duration,
        "config": asdict(config),
        "learning_rate": lr,
        "batch_size": batch_size,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "patience": args.patience,
        "dataset": {
            "train_rows": int(train_frame.shape[0]),
            "valid_rows": int(valid_frame.shape[0]),
            "test_rows": int(test_frame.shape[0]) if test_frame is not None else 0,
        },
    }
    if test_metrics is not None:
        metrics_out["test"] = test_metrics

    save_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        config=config,
        temp_mean=temp_mean,
        temp_std=temp_std,
        metrics={
            **metrics,
            **({"test_mse": float(test_metrics["mse"]), "test_mae": float(test_metrics["mae"]), "test_r2": float(test_metrics["r2"])} if test_metrics else {}),
        },
        duration_seconds=duration,
        training_params={
            "learning_rate": lr,
            "batch_size": batch_size,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "patience": args.patience,
        },
        epoch=int(metrics.get("last_epoch", 0.0)),
        optimizer_state_dict=optimizer.state_dict(),
    )

    # Convenience copy for "best model" naming.
    try:
        best_named = args.output_dir / f"{MODEL_NAME}_best.pt"
        if checkpoint_path.exists():
            best_named.write_bytes(checkpoint_path.read_bytes())
            print_report(f"Saved best model copy: {best_named}")
    except Exception as e:  # pragma: no cover
        print_report(f"Warning: failed to write best model copy: {e}")

    metrics_path = args.output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics_out, fp, indent=2)
    print_report(f"Stored metrics at {metrics_path}")
    report_fp.close()


if __name__ == "__main__":
    main()


