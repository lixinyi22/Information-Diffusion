# Introduction

This repository contains the datasets and associated code used in the paper: A Multi-Channel Integrated Framework for
Information Diffusion

In this paper, we propose MCID, a Multi-Channel integrated framework for Information Diffusion. MCID introduces information-side modeling through three complementary channels: a user–user relation channel, an information interactive channel, and an information contextual channel. By integrating structural, interactive, and contextual perspectives, MCID enables comprehensive representation learning beyond traditional user-centric approaches.

# Dataset

The dataset in `./data` contains the Twitter, Weibo, and Zhihu datasets that we used in the experiments for evaluation. The basic information of the datasets is as follows.

|             | Twitter | Weibo | Zhihu  |
| ----------- | ------- | ----- | ------  |
| #user       | 8,292   | 6,512 | 30,000  |
| #cascades   | 2,798   | 2,267 | 3,326   |
| #Links      | 86,768  | -     | -       |
| Avg. Length | 5.39    | 43.73 | 12.29   |

Each dataset contains all the original cascades, the information content embedding, cascade ID dict and user ID dict.

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
Information-Diffusion/
├── data/                          # Dataset directory
│   ├── Twitter/                   
│   ├── Weibo/                     
│   └── Zhihu/                     
├── src/                           # Source code directory
│   ├── model/                     # Model implementation
│   │   ├── layers.py              # Neural network layers (Transformer, etc.)
│   │   └── model.py               # Main model architecture (MCID)
│   ├── utils/                     # Utility functions
│   │   ├── Constants.py           # Constants and configurations
│   │   ├── DataConstruct.py       # Data loading and preprocessing
│   │   ├── GraphConstruct.py      # Graph construction utilities
│   │   ├── Metrics.py             # Evaluation metrics
│   │   └── Optim.py               # Optimizer utilities
│   └── run.py                     # Main entry point for training and evaluation
├── README.md                      # Project documentation
└── requirements.txt               # Python dependencies
```
