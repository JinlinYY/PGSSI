# PGSSI



This repository contains the code and datasets used in the manuscript:

> **Physics-Guided 3D Solute-Solvent Interaction Framework for Infinite-Dilution Activity Coefficient Prediction**.

PGSSI predicts temperature-dependent infinite-dilution activity coefficients, `log-gamma`, from solute SMILES, solvent SMILES, and temperature. The model builds a joint 3D solute-solvent representation with explicit intermolecular contact edges, learns intramolecular and cross-molecular interactions, and maps the learned representation to a thermodynamically structured inverse-temperature response.


## Repository Structure

```text
PGSSI-github/
|-- dataset/
|   |-- all/                         # Full merged dataset and train/valid/test splits
|   |-- wu_et_al/                    # Wu et al. benchmark split
|   `-- wang_et_al/                  # IDAC2026 external test data
|-- src/
|   |-- models/PGSSI/                # Model architecture, data processing, training, physics loss
|   |-- benchmark/                   # Wu2004 / IDAC2026 benchmark script
|   |-- ablation/                    # Ablation experiment scripts
|   |-- isothermal/                  # Isothermal split experiments
|   |-- interpretability/            # PGSSI interpretation and feature importance analysis
|   |-- dataset_split/               # Pair-grouped dataset splitting
|   |-- result_plot/                 # Plotting utilities
|   `-- continuous interpolation and extrapolation/
|-- cache/                           # Generated geometry caches
`-- README.md
```

## Data

The main CSV files use the following columns:

- `Solute_SMILES`: solute molecule SMILES
- `Solvent_SMILES`: solvent molecule SMILES
- `T`: temperature in Celsius
- `log-gamma`: target infinite-dilution activity coefficient in log scale

Included datasets:

| Dataset | File | Samples |
| --- | --- | ---: |
| Full merged dataset | `dataset/all/all_merged.csv` | 39,840 |
| Full train split | `dataset/all/all_merged_train.csv` | 31,853 |
| Full validation split | `dataset/all/all_merged_valid.csv` | 4,097 |
| Full test split | `dataset/all/all_merged_test.csv` | 3,890 |
| Wu et al. train/valid/test | `dataset/wu_et_al/` | 21,284 total |
| IDAC2026 external data | `dataset/wang_et_al/IDAC_2026_dataset.csv` | 18,556 |

## Environment

The project requires PyTorch, PyTorch Geometric, RDKit, and common scientific Python packages. Because PyTorch/PyG installation depends on your CUDA version, install those packages using the commands recommended for your platform.

Example setup:

```bash
conda create -n pgssi python=3.9 -y
conda activate pgssi

# Install RDKit.
conda install -c conda-forge rdkit -y

# Install PyTorch for your CPU/CUDA environment.
# See: https://pytorch.org/get-started/locally/

# Install PyTorch Geometric and compiled extensions for your PyTorch/CUDA version.
# See: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

pip install numpy pandas scikit-learn matplotlib tqdm tabulate
```

The first PGSSI run generates 3D molecular-pair geometry caches under `cache/`. This can take time, but later runs reuse the cached data.

## Quick Start

Run all commands from the repository root.

Train PGSSI on the full dataset split:

```bash
python src/models/PGSSI/PGSSI_train.py \
  --train-path dataset/all/all_merged_train.csv \
  --valid-path dataset/all/all_merged_valid.csv \
  --test-path dataset/all/all_merged_test.csv \
  --run-dir runs/pgssi_full \
  --cache-dir cache/pgssi_full \
  --model-name PGSSI \
  --n-epochs 300 \
  --batch-size 32
```

For a smaller smoke test:

```bash
python src/models/PGSSI/PGSSI_train.py \
  --train-path dataset/all/subset10/all_merged_train.csv \
  --valid-path dataset/all/subset10/all_merged_valid.csv \
  --test-path dataset/all/subset10/all_merged_test.csv \
  --run-dir runs/pgssi_subset10 \
  --cache-dir cache/pgssi_subset10 \
  --n-epochs 5 \
  --batch-size 16
```

Training outputs are written to the selected `runs/` directory, including model checkpoints, training curves, prediction CSV files, and JSON metric summaries.

## Benchmark Experiments

Run the Wu2004 / IDAC2026 benchmark:

```bash
python src/benchmark/run_wu_wang_benchmark.py \
  --run-dir runs/pgssi_wu_benchmark \
  --cache-dir cache/pgssi_wu_benchmark \
  --n-epochs 300 \
  --batch-size 32
```



## Ablation Studies

Run the ablation grid:

```bash
python src/ablation/run_ablation.py \
  --train-path dataset/all/all_merged_train.csv \
  --valid-path dataset/all/all_merged_valid.csv \
  --test-path dataset/all/all_merged_test.csv \
  --output-dir runs/pgssi_ablation \
  --epochs 200
```



## Additional Analyses

Isothermal experiments:

```bash
python src/isothermal/run_isothermal.py \
  --output-dir runs/pgssi_isothermal \
  --epochs 50
```

Prediction result plots:

```bash
python src/result_plot/plot_pgssi_results.py \
  --input runs/pgssi_full/all_merged_train_PGSSI_all_merged_test_predictions.csv \
  --output-dir runs/pgssi_full/figures
```

Interpretability and feature-importance scripts are located in `src/interpretability/`. These scripts expect a trained PGSSI checkpoint and a compatible test CSV.

