# PGSSI



This repository contains the code and datasets used in the manuscript:

> **Physics-Guided 3D Solute-Solvent Interaction Framework for Infinite-Dilution Activity Coefficient Prediction**.

PGSSI predicts temperature-dependent infinite-dilution activity coefficients, from solute SMILES, solvent SMILES, and temperature. The model builds a joint 3D solute-solvent representation with explicit intermolecular contact edges, learns intramolecular and cross-molecular interactions, and maps the learned representation to a thermodynamically structured inverse-temperature response.


![PGSSI framework](https://github.com/JinlinYY/PGSSI/blob/main/method.png)

## Revision Experiments

The repository includes the additional analyses conducted during manuscript revision:

| Experiment                                            | Script                                                       | Main outputs                                                 |
| ----------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Dataset composition and quality analysis              | `src/dataset_analysis/analyze_dataset_composition.py`        | Dataset figures, summary tables, and duplicate/conflict records |
| Isomerism subset and pair-grouped 5-fold CV           | `src/Isomerism_experiment/run_isomerism_cv.py`               | Subset analysis, fold files, Full PGSSI and topology-only results |
| Interaction-type ablation with pair-grouped 5-fold CV | `src/Interaction_type_ablation/run_interaction_type_ablation.py` | Per-fold and mean +/- std results for six interaction masks  |

All reported MAE, RMSE, and R2 values are calculated on `log-gamma`.

## Repository Structure

```text
PGSSI/
|-- dataset/
|   |-- all/                         # Full merged dataset and train/valid/test splits
|   |-- Isomerism_experiment/        # Isomerism analysis and pair-grouped folds
|   |-- Interaction_type_ablation/   # Pair-grouped folds for interaction ablation
|   |-- wu_et_al/                    # Wu et al. benchmark split
|   `-- wang_et_al/                  # IDAC2026 external test data
|-- src/
|   |-- models/PGSSI/                # Model architecture, data processing, training, physics loss
|   |-- benchmark/                   # Wu2004 / IDAC2026 benchmark script
|   |-- ablation/                    # Ablation experiment scripts
|   |-- dataset_analysis/            # Dataset composition and quality analysis
|   |-- Isomerism_experiment/        # Isomerism subset and 5-fold CV
|   |-- Interaction_type_ablation/   # Interaction-feature ablation and results
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

| Dataset                    | File                                       |      Samples |
| -------------------------- | ------------------------------------------ | -----------: |
| Full merged dataset        | `dataset/all/all_merged.csv`               |       39,840 |
| Full train split           | `dataset/all/all_merged_train.csv`         |       31,853 |
| Full validation split      | `dataset/all/all_merged_valid.csv`         |        4,097 |
| Full test split            | `dataset/all/all_merged_test.csv`          |        3,890 |
| Wu et al. train/valid/test | `dataset/wu_et_al/`                        | 21,284 total |
| IDAC2026 external data     | `dataset/wang_et_al/IDAC_2026_dataset.csv` |       18,556 |

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

Alternatively, after installing a CUDA-compatible PyTorch build, install the remaining project dependencies with:

```bash
pip install -r requirements.txt
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


## Dataset Composition Analysis

Generate the dataset composition figure, publication tables, and duplicate/conflict records:

```bash
python src/dataset_analysis/analyze_dataset_composition.py \
  --input-csv dataset/all/all_merged.csv \
  --output-dir src/dataset_analysis
```

The analysis reports 39,840 samples, 559 unique solutes, 695 unique solvents, and 13,222 unique solute-solvent pairs. It also summarizes source composition, temperature and `log-gamma` coverage, duplicate/conflicting records, isomeric or stereochemical markers, H-bond-rich systems, and water-containing systems.


## Isomerism Subset and 5-Fold Cross-Validation

The isomerism workflow identifies stereochemical/isomeric samples using RDKit canonical and isomeric SMILES, with `@`, `/`, and `\` SMILES markers as a fallback. It then creates solute-solvent-pair-grouped folds and evaluates Full PGSSI against the `--topology-only` baseline on both the full dataset and the isomerism subset.

Prepare the subset and fold files without training:

```bash
python src/Isomerism_experiment/run_isomerism_cv.py --prepare-only
```

Run the complete experiment:

```bash
python src/Isomerism_experiment/prebuild_3d_cache.py

python src/Isomerism_experiment/run_isomerism_cv.py \
  --epochs 300 \
  --batch-size 32 \
  --quiet-progress
```

The complete per-fold results and paper-ready tables are available in [`src/Isomerism_experiment/`](src/Isomerism_experiment/).

## Interaction-Type Ablation

This experiment masks selected handcrafted intermolecular indicators while retaining distance/radial, Lennard-Jones, Coulomb, and charge-based features. The six settings are Full PGSSI, no H-bond tendency, no aromatic/pi features, no dipole alignment/opposition, no hydrophobic/polar features, and no explicit interaction-type indicators.

```bash
python src/Interaction_type_ablation/run_interaction_type_ablation.py \
  --data-path dataset/all/all_merged.csv \
  --folds-dir dataset/Interaction_type_ablation/folds \
  --output-dir src/Interaction_type_ablation/outputs_5fold \
  --cache-dir src/Interaction_type_ablation/cache/datasets \
  --pair-cache-dir src/Interaction_type_ablation/cache/pair_graphs \
  --epochs 300 \
  --early-stopping-patience 50 \
  --batch-size 32
```

The complete summary, per-fold metrics, conclusions, and LaTeX tables are available in [`src/Interaction_type_ablation/outputs_5fold/`](src/Interaction_type_ablation/outputs_5fold/).

## Additional Analyses

Isothermal experiments:

```bash
python src/isothermal/run_isothermal.py \
  --output-dir runs/pgssi_isothermal \
  --epochs 50
```

Interpretability and feature-importance scripts are located in `src/interpretability/`. These scripts expect a trained PGSSI checkpoint and a compatible test CSV.

