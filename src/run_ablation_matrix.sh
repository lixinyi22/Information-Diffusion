#!/bin/bash
# 多 GPU 消融实验队列调度（兼容在 screen/tmux 会话中运行）

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAILED_DIR="$SCRIPT_DIR/failed_jobs"
mkdir -p "$FAILED_DIR"
FAILED_LOG="$FAILED_DIR/failed_jobs_$(date +%y-%m-%d_%H-%M-%S).log"

# ====== 实验矩阵（按需改这里） ======
# 固定单卡配置（去掉自动选卡）
FIXED_GPU=0
GPU_LIST=("$FIXED_GPU")

DATASETS=(Weibo)
ABLATIONS=(full) # only_structural only_interactive only_contextual
FUSION_MODES=(sum concat) #
# 对应 run.py --fusion_weight_mode：manual（固定 fusion_w_*）| adaptive（可学习 softmax×3）| disable（恒为 1）
FUSION_WEIGHT_MODES=(manual)

SELF_GATE_SETTINGS=(enabled) # enabled=保留self gate, disabled=加 --disable_self_gate
USER_CONTENT_SETTINGS=(disabled) # enabled=加 --use_history_user_content
SEEDS=(2026)

# 与 run.py --fusion_w_g / --fusion_w_i / --fusion_w_c 对应（手动融合权重，默认 1.0）
# 每项为三个浮点数："wg wi wc"，一行一组；可写多行做网格外的自定义组合
FUSION_W_TRIPLETS=(
  "1.0 2.0 1.0"
)

# 训练超参（可按需覆盖）
# num_seq_layers / num_mm_layers / num_hg_layers 未传入，使用 run.py 默认值（1 / 2 / 2）
KNN_K=10
EPOCH=200
PATIENCE=10
EXTRA_ARGS=""
# 例如：EXTRA_ARGS="-batch_size 16 -lr 5e-4"
# ===================================

JOBS=()
for dataset in "${DATASETS[@]}"; do
  for ablation in "${ABLATIONS[@]}"; do
    for fusion_mode in "${FUSION_MODES[@]}"; do
      for fusion_weight_mode in "${FUSION_WEIGHT_MODES[@]}"; do
        for self_gate_setting in "${SELF_GATE_SETTINGS[@]}"; do
          for user_content_setting in "${USER_CONTENT_SETTINGS[@]}"; do
            for seed in "${SEEDS[@]}"; do
              for fusion_w_triplet in "${FUSION_W_TRIPLETS[@]}"; do
                read -r fusion_w_g fusion_w_i fusion_w_c <<< "$fusion_w_triplet"
                JOBS+=("${dataset}|${ablation}|${fusion_mode}|${fusion_weight_mode}|${self_gate_setting}|${user_content_setting}|${seed}|${fusion_w_g}|${fusion_w_i}|${fusion_w_c}")
              done
            done
          done
        done
      done
    done
  done
done

TOTAL_JOBS=${#JOBS[@]}
if [[ "$TOTAL_JOBS" -eq 0 ]]; then
  echo "没有可执行任务，退出。"
  exit 0
fi

echo "======== 多 GPU 消融任务开始: total=${TOTAL_JOBS} ($(date)) ========"
echo "失败任务将记录到: ${FAILED_LOG}"

declare -A PID_TO_GPU
declare -A PID_TO_JOB
RUNNING_PIDS=()
NEXT_JOB_INDEX=0
DONE_COUNT=0
FAIL_COUNT=0

launch_job_on_gpu() {
  local gpu="$1"
  local job="$2"
  IFS='|' read -r dataset ablation fusion_mode fusion_weight_mode self_gate_setting user_content_setting seed fusion_w_g fusion_w_i fusion_w_c <<< "$job"
  local self_gate_args=()
  local user_content_args=()
  if [[ "$self_gate_setting" == "disabled" ]]; then
    self_gate_args+=(--disable_self_gate)
  fi
  if [[ "$user_content_setting" == "enabled" ]]; then
    user_content_args+=(--use_history_user_content)
  fi

  echo "---- Launch: dataset=${dataset} ablation=${ablation} fusion_mode=${fusion_mode} fusion_weight_mode=${fusion_weight_mode} fusion_w=(g=${fusion_w_g},i=${fusion_w_i},c=${fusion_w_c}) self_gate=${self_gate_setting} user_content=${user_content_setting} seed=${seed} gpu=${gpu} ($(date)) ----"
  python "$SCRIPT_DIR/run.py" \
    --data "$dataset" \
    --ablation_mode "$ablation" \
    --fusion_mode "$fusion_mode" \
    --fusion_weight_mode "$fusion_weight_mode" \
    --fusion_w_g "$fusion_w_g" \
    --fusion_w_i "$fusion_w_i" \
    --fusion_w_c "$fusion_w_c" \
    "${self_gate_args[@]}" \
    "${user_content_args[@]}" \
    -seed "$seed" \
    -knn_k "$KNN_K" \
    -epoch "$EPOCH" \
    -patience "$PATIENCE" \
    --gpu "$gpu" \
    $EXTRA_ARGS &

  local pid=$!
  PID_TO_GPU["$pid"]="$gpu"
  PID_TO_JOB["$pid"]="$job"
  RUNNING_PIDS+=("$pid")
  # run.py 的 save_dir 使用秒级时间戳；轻微错峰可降低目录同名风险
  sleep 1
}

remove_pid_from_running() {
  local target_pid="$1"
  local new_running=()
  local pid
  for pid in "${RUNNING_PIDS[@]}"; do
    if [[ "$pid" != "$target_pid" ]]; then
      new_running+=("$pid")
    fi
  done
  RUNNING_PIDS=("${new_running[@]}")
}

while [[ "$DONE_COUNT" -lt "$TOTAL_JOBS" ]]; do
  # 尽可能填满空闲 GPU
  while [[ "$NEXT_JOB_INDEX" -lt "$TOTAL_JOBS" && "${#RUNNING_PIDS[@]}" -lt "${#GPU_LIST[@]}" ]]; do
    gpu="${GPU_LIST[${#RUNNING_PIDS[@]}]}"
    launch_job_on_gpu "$gpu" "${JOBS[$NEXT_JOB_INDEX]}"
    NEXT_JOB_INDEX=$((NEXT_JOB_INDEX + 1))
  done

  # 没有运行中的任务（理论上不该出现）
  if [[ "${#RUNNING_PIDS[@]}" -eq 0 ]]; then
    break
  fi

  # 轮询是否有任务结束
  finished_pid=""
  exit_code=0
  for pid in "${RUNNING_PIDS[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"
      exit_code=$?
      finished_pid="$pid"
      break
    fi
  done

  # 若还没有任务结束，短暂休眠再轮询
  if [[ -z "$finished_pid" ]]; then
    sleep 5
    continue
  fi

  done_job="${PID_TO_JOB[$finished_pid]}"
  done_gpu="${PID_TO_GPU[$finished_pid]}"
  IFS='|' read -r dataset ablation fusion_mode fusion_weight_mode self_gate_setting user_content_setting seed fusion_w_g fusion_w_i fusion_w_c <<< "$done_job"
  DONE_COUNT=$((DONE_COUNT + 1))

  if [[ "$exit_code" -ne 0 ]]; then
    FAIL_COUNT=$((FAIL_COUNT + 1))
    echo "$(date '+%F %T')|gpu=${done_gpu}|dataset=${dataset}|ablation=${ablation}|fusion_mode=${fusion_mode}|fusion_weight_mode=${fusion_weight_mode}|fusion_w_g=${fusion_w_g}|fusion_w_i=${fusion_w_i}|fusion_w_c=${fusion_w_c}|self_gate=${self_gate_setting}|user_content=${user_content_setting}|seed=${seed}|exit_code=${exit_code}" >> "$FAILED_LOG"
    echo "!!!! Failed: dataset=${dataset} ablation=${ablation} fusion_mode=${fusion_mode} fusion_weight_mode=${fusion_weight_mode} fusion_w=(g=${fusion_w_g},i=${fusion_w_i},c=${fusion_w_c}) self_gate=${self_gate_setting} user_content=${user_content_setting} seed=${seed} gpu=${done_gpu} exit=${exit_code}"
  else
    echo "++++ Done: dataset=${dataset} ablation=${ablation} fusion_mode=${fusion_mode} fusion_weight_mode=${fusion_weight_mode} fusion_w=(g=${fusion_w_g},i=${fusion_w_i},c=${fusion_w_c}) self_gate=${self_gate_setting} user_content=${user_content_setting} seed=${seed} gpu=${done_gpu}"
  fi

  remove_pid_from_running "$finished_pid"

  # 如果还有待执行任务，复用刚释放的 GPU 立即启动下一个
  if [[ "$NEXT_JOB_INDEX" -lt "$TOTAL_JOBS" ]]; then
    launch_job_on_gpu "$done_gpu" "${JOBS[$NEXT_JOB_INDEX]}"
    NEXT_JOB_INDEX=$((NEXT_JOB_INDEX + 1))
  fi
done

echo "======== 多 GPU 消融任务结束: done=${DONE_COUNT}/${TOTAL_JOBS}, failed=${FAIL_COUNT} ($(date)) ========"
if [[ "$FAIL_COUNT" -gt 0 ]]; then
  echo "失败任务详情: ${FAILED_LOG}"
fi
