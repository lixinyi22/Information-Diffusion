# Introduction

This repository contains the datasets and associated code used in the paper: An Unified Multi-Channel Framework for Information Diffusion.

# Dataset

The dataset in `./data` contains the Twitter, Weibo, Zhihu and KuaiRand datasets that we used in the experiments for evaluation.
Each dataset contains all the original cascades, the information content embedding, cascade ID dict and user ID dict.

## Data Format

Each dataset folder (e.g., `./data/Twitter/`) contains the following files:

- **Cascade files**:
  - `cascade.txt`, `cascadevalid.txt`, `cascadetest.txt`: Training, validation, and test cascade sequences. Each line represents a cascade in the format `user_id,timestamp user_id,timestamp ...`, where users are ordered by their participation time in the diffusion process.
  - `cascade_id.txt`, `cascadevalid_id.txt`, `cascadetest_id.txt`: Corresponding information IDs for each cascade.

- **Index mapping files** (pickle format):
  - `u2idx.pickle`: Dictionary mapping user IDs to indices
  - `idx2u.pickle`: Dictionary mapping indices to user IDs

- **Content embedding file**:
  - `id2embedding.pickle`: Dictionary mapping information IDs to their pre-computed content embeddings

The data loading utilities in `src/utils/DataConstruct.py` automatically handle the parsing and preprocessing of these files during training and evaluation.

# CODE

## Dependencies

Install the dependencies:

+ Python (>=3.10)
+ PyTorch (>=2.5.1)
+ NumPy (>=2.2.1)
+ Pandas (>=2.2.3)
+ Scipy (>=1.14.1)
+ scikit-learn (>=1.6.0)
+ torch-geometric (>=2.6.1)
+ tqdm (>=4.67.1)

```bash
# create virtual environment
conda create --name MCID python=3.12

# activate environment
conda activate MCID


# install other dependencies
pip install -r requirements.txt
```

## Getting Started
To run the implementation of MCID, you can use the following code. More running options are described in the codes.

```bash
python run.py
```

# Folder Structure

```
Information_Diffusion/
├── data/                          # Dataset directory
│   ├── Twitter_0607/              # Twitter cascade data
│   ├── Weibo/                     # Weibo cascade data
│   ├── Zhihu/                     # Zhihu cascade data
│   ├── KuaiRand/                  # KuaiRand cascade data
├── src/                       # Source code directory
│   ├── model/                 # Model implementation
│   │   ├── layers.py          # Neural network layers
│   │   ├── model.py           # Main MCID architecture
│   ├── utils/                 # Utility functions
│   │   ├── Constants.py       # Constants and configurations
│   │   ├── DataConstruct.py   # Data loading and preprocessing
│   │   ├── GraphConstruct.py  # Graph construction utilities
│   │   ├── Metrics.py         # Evaluation metrics
│   │   ├── Optim.py           # Optimizer utilities
│   │   └── compute_stats.py   # Dataset statistics utilities
│   ├── run.py                 # Main entry point for training and evaluation
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies
```