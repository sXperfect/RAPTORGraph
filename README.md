# RAPTORGraph: Causal Representation Learning for Single-Cell Perturbation Response Modeling

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Placeholder-orange)](LICENSE)

This repository contains the official implementation of **RAPTORGraph**, a causal representation learning framework for modeling single-cell perturbation responses.

## Table of Contents

- [Overview](#overview)
- [Abstract](#abstract)
- [Installation](#installation)
- [Data](#data)
- [Models](#models)
- [Usage](#usage)
- [Reproducibility](#reproducibility)
- [Repository Structure](#repository-structure)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## Abstract

Experiments involving the perturbation of individual cells are central to understanding cellular mechanisms and, can accelerate drug discovery and improve
therapy. Causal representation learning (CRL) allows us to uncover the latent
factors that regulate biological systems and predict the impact of novel perturbations. Unfortunately, existing methods fail to address intervention spillover in
a closed-world setting where intervention targets are known a priori, such as in
Perturb-seq experiments, due to their reliance on dense encoders. Furthermore,
incorporating curated biological pathways into the model imposes a confirmatory
bias, forcing it to explain the data through preexisting pathways and reducing the set
of hypotheses the model can explore while discarding novel signals that lie outside
the annotated pathways. In this work, we introduce RAPTORGraph, a β-VAE with
a GraphPathway encoder that explicitly models complex gene-to-gene interactions
within learned pathways. Moreover, our model’s preconditioning isolates the influence of perturbed genes, yielding clean, single-node latent interventions required
for identifiable causal discovery and eliminating spillover. Finally, we train the
model on data preprocessed with optimal-transport alignment, which guarantees a
well-defined mapping between control and perturbed samples and further stabilizes
the learned latent representations. We demonstrate that RAPTORGraph improves
state-of-the-art performance on downstream analyses of unseen perturbations, such
as non-additive interactions, while outperforming other approaches on objective
metrics, such as MSE and MMD.

## Installation

### Prerequisites

- Python 3.11 or higher
- Conda or Miniconda (recommended for environment management)

### Creating the Conda Environment

```bash
# Create a new conda environment with Python 3.11
conda create -n raptorgraph python=3.11

# Activate the environment
conda activate raptorgraph

# Install required packages from requirements.txt
pip install -r requirements.txt
```

## Data

The datasets used in this study are publicly available. To download and preprocess the data, please follow the instructions in the `data/datasets/` directory. A placeholder file `THIS_IS_DATASETS` is there to indicate where the datasets should be.

### Downloading the Norman et al. 2019 Dataset

The default dataset used in our experiments is the Norman et al. 2019 Perturb-seq dataset. To download and set it up:

```bash
# Navigate to the datasets directory
cd data/datasets

# Download the CPA binaries archive
wget https://dl.fbaipublicfiles.com/dlp/cpa_binaries.tar

# Extract the archive
tar -xvf cpa_binaries.tar

# Move the Norman2019_raw.h5ad file to the current directory
mv datasets/Norman2019_raw.h5ad .

# Clean up temporary files
rm -rf datasets pretrained_models cpa_binaries.tar
```

After running these commands, the `Norman2019_raw.h5ad` file (approximately 762MB) will be placed directly in the `data/datasets/` directory, which is where the code expects to find it.

### Data Verification

You can verify that the dataset was correctly downloaded by checking that the file exists:

```bash
ls -la data/datasets/Norman2019_raw.h5ad
```

This should show a file of approximately 762MB.

## Models

The trained models are available in the `data/models/` directory. A placeholder file `THIS_IS_MODELS` is there to indicate where the models should be.

## Usage

### Cache the Dataset

Before training the model, we recommend to cache the dataset by running RAPTORGraph in `cache` mode after downloading the dataset:

```bash
python run_raptorgraph.py args.mode=cache +exp/run_raptorgraph=run_exp
```

### Basic Usage

To run RAPTORGraph with default parameters:

```bash
python run_raptorgraph.py args.mode=run +exp/run_raptorgraph=run_exp
```

### Configuration

RAPTORGraph uses Hydra for configuration management. Experiments are defined in YAML files located in the `configs/exp/` directory. To run with a specific experiment configuration:

```bash
python run_raptorgraph.py args.mode=run +exp/run_raptorgraph=your_experiment_config
```

### Key Parameters

- `args.dry_run`: Run without logging and persistent storage for quick testing
- `args.mode`: Set to `run` for training/evaluation mode and `cache` to cache the dataset.
- `pl.trainer.max_epochs`: Maximum number of training epochs
- `pl.trainer.accelerator`: Hardware accelerator ("cpu", "gpu")

### Example Command

```bash
# Run with GPU acceleration for 100 epochs
python run_raptorgraph.py \
  args.mode=run \
  +exp/run_raptorgraph=your_experiment_config
  args.dry_run=False \
  pl.trainer.max_epochs=100 \
  pl.trainer.accelerator=gpu
```

## Reproducibility

For reproducibility of our experimental results:

1. Create the conda environment as described above
2. Download the required datasets using the instructions above
3. Run the experiment configurations used in our paper:

```bash
python run_raptorgraph.py args.mode=cache +exp/run_raptorgraph=run_exp
python run_raptorgraph.py args.mode=run +exp/run_raptorgraph=run_exp
  ```

## Contributing

We welcome contributions to RAPTORGraph. If you would like to contribute, please open an issue or submit a pull request.

## Citation

If you use RAPTORGraph in your research, please cite our paper:

```
@article{raptorgraph2026,
  title={{RAPTORGraph: Causal Representation Learning for Single-Cell Perturbation Response Modeling}},
  author={Anonymous},
  journal={Anonymous}
}
```

## License

This project is licensed under a placeholder license for double-blind review. The actual license will be added upon acceptance of the paper. See the [LICENSE](LICENSE) file for details.

## Contact

For questions about this codebase, please open an issue on this repository. For inquiries about the paper, please contact the corresponding author through the conference's submission system during the review period.
