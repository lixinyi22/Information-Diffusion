#!/bin/bash
# 连续测试不同 kNN 的 k 值，并对每个 k 使用多个随机种子。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# GPU 编号（字符串，传给 run.py 的 --gpu，对应 CUDA_VISIBLE_DEVICES）
GPU="1"

# datasets to run
DATASETS=(Weibo)

# k values for kNN graph (run.py uses "-knn_k")
K_VALUES=(100)

# random seeds (run.py uses "-seed")
SEEDS=(42)

for dataset in "${DATASETS[@]}"; do
  echo "======== 数据集开始: ${dataset}  ($(date)) ========"
  for k in "${K_VALUES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo "---- Test: dataset=${dataset}  knn_k=${k}  seed=${seed}  ($(date)) ----"
      python "$SCRIPT_DIR/run.py" --data "$dataset" -knn_k "$k" -seed "$seed" --gpu "$GPU"
    done
  done
  echo "======== 数据集完成: ${dataset}  ($(date)) ========"
done

echo "======== 全部测试完成 ($(date)) ========"
