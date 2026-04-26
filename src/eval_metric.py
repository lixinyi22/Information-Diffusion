import pandas as pd
import ast
import numpy as np
import sys
import os
import pickle
from collections import Counter, defaultdict, deque
from scipy import stats

import utils.Constants as Constants

def analyze_diff(test_path, test2_path, ks=[10, 50, 100]):
    df1 = pd.read_csv(test_path)
    df2 = pd.read_csv(test2_path)
    assert len(df1) == len(df2), "两个文件的样本数不一致"
    total = len(df1)
    print("total: ", total)

    df1['pred_results'] = df1['pred_results'].apply(ast.literal_eval)
    df2['pred_results'] = df2['pred_results'].apply(ast.literal_eval)

    for k in ks:
        test_right = 0
        test2_right = 0
        test2_right_test_wrong = 0

        for i in range(total):
            gt = df1.iloc[i]['ground_truth']
            pred1_topk = df1.iloc[i]['pred_results'][:k]
            pred2_topk = df2.iloc[i]['pred_results'][:k]
            in_test = gt in pred1_topk
            in_test2 = gt in pred2_topk
            if in_test:
                test_right += 1
            if in_test2:
                test2_right += 1
            if in_test2 and not in_test:
                test2_right_test_wrong += 1

        print(f"k={k}:")
        #print(f"中间结果正确样本数: {test2_right}")
        print(f"这些错误样本占所有中间结果正确样本的{test2_right_test_wrong/test2_right:.4f}")
        print(f"如果这类错误都能修正，最终命中率可提升到:{(test_right + test2_right_test_wrong)/total:.4f}，命中率提升幅度为: {((test_right + test2_right_test_wrong)/total - test_right/total)/(test_right/total):.4f}")
        print(f"当中间结果错误时的最终错误率{((total-test_right-test2_right_test_wrong)/(total-test2_right)):.4f}")
        print(f"此类错误影响了{(test2_right_test_wrong/total):.4f}的总样本，占所有最终错误样本的{(test2_right_test_wrong/(total-test_right)):.4f}")


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This matches the implementation in Metrics.py

    Args:
        actual: list of ground truth values (usually contains one element)
        predicted: list of predicted values
        k: maximum number of predicted elements
    """
    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    # 与Metrics.py保持一致: return score / min(len(actual), k)
    return score / min(len(actual), k)


def compute_metrics_from_predictions(predictions_dict, k_list=[10, 50, 100]):
    """
    从predictions_dict计算每个样本的metric

    Args:
        predictions_dict: dict with 'topk_predictions' and 'y_true'
        k_list: list of k values

    Returns:
        metrics_dict: dict with metric names as keys and lists of per-sample scores as values
    """
    topk_predictions = predictions_dict['topk_predictions']  # list of lists
    y_true = predictions_dict['y_true']  # list

    assert len(topk_predictions) == len(y_true), "预测和真实标签数量不一致"

    metrics_dict = {}
    for k in k_list:
        metrics_dict[f'hits@{k}'] = []
        metrics_dict[f'map@{k}'] = []

    for pred_list, true_label in zip(topk_predictions, y_true):
        for k in k_list:
            # 获取topk预测
            topk = pred_list[:k]

            # 计算hits@k
            hits = 1.0 if true_label in topk else 0.0
            metrics_dict[f'hits@{k}'].append(hits)

            # 计算map@k (与Metrics.py保持一致，传入列表)
            map_score = apk([true_label], topk, k)
            metrics_dict[f'map@{k}'].append(map_score)

    return metrics_dict


def t_test_comparison(predictions_dict1, predictions_dict2, k_list=[10, 50, 100], alpha_list=[0.01, 0.05]):
    """
    对两个baseline的预测结果进行配对t检验

    Args:
        predictions_dict1: 第一个baseline的overall_predictions_dict
        predictions_dict2: 第二个baseline的overall_predictions_dict
        k_list: 评价指标中的k值列表
        alpha: 显著性水平，默认0.05

    Returns:
        results: dict containing t-test results for each metric
    """
    # 计算每个样本的metric
    metrics1 = compute_metrics_from_predictions(predictions_dict1, k_list)
    metrics2 = compute_metrics_from_predictions(predictions_dict2, k_list)

    for k in k_list:
        for metric_name in [f'hits@{k}', f'map@{k}']:
            scores1 = np.array(metrics1[metric_name])
            scores2 = np.array(metrics2[metric_name])

            # 计算均值
            mean1 = np.mean(scores1)
            mean2 = np.mean(scores2)
            mean_diff = mean2 - mean1

            # 配对t检验
            t_statistic, p_value = stats.ttest_rel(scores1, scores2)

            # 计算效果量 (Cohen's d)
            diff = scores2 - scores1
            cohens_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-8)

            # 打印结果
            print(f"\n{metric_name}:")
            print(f"  Baseline 1 均值: {mean1:.6f}")
            print(f"  Baseline 2 均值: {mean2:.6f}")
            print(f"  均值差异: {mean_diff:.6f} ({'提升' if mean_diff > 0 else '下降'})")
            print(f"  t统计量: {t_statistic:.6f}")
            print(f"  p值: {p_value:.6f}")
             # 判断显著性
            for alpha in alpha_list:
                is_significant = p_value < alpha
                significance = f"{alpha}显著" if is_significant else f"{alpha}不显著"
                print(f"  {significance}")
            print(f"  Cohen's d (效果量): {cohens_d:.6f}")

            # 解释效果量
            if abs(cohens_d) < 0.2:
                effect_size = "小"
            elif abs(cohens_d) < 0.5:
                effect_size = "中"
            elif abs(cohens_d) < 0.8:
                effect_size = "大"
            else:
                effect_size = "非常大"
            print(f"  效果量: {effect_size}")


def _try_align_by_y_true(reference_dict, target_dict):
    """
    在样本集合一致但顺序不一致时，按 y_true 对 target 做稳定重排，使其与 reference 对齐。
    """
    y_true_ref = reference_dict['y_true']
    y_true_target = target_dict['y_true']

    if len(y_true_ref) != len(y_true_target):
        return None, "样本数不一致，无法对齐"

    if y_true_ref == y_true_target:
        return target_dict, "y_true 顺序已一致，无需对齐"

    if Counter(y_true_ref) != Counter(y_true_target):
        return None, "两侧 y_true 的标签分布不同，无法按标签重排"

    idx_queues = defaultdict(deque)
    for idx, label in enumerate(y_true_target):
        idx_queues[label].append(idx)

    aligned_y_true = []
    aligned_topk = []
    for label in y_true_ref:
        if not idx_queues[label]:
            return None, "重排过程中出现空队列，无法继续对齐"
        idx = idx_queues[label].popleft()
        aligned_y_true.append(y_true_target[idx])
        aligned_topk.append(target_dict['topk_predictions'][idx])

    aligned_dict = dict(target_dict)
    aligned_dict['y_true'] = aligned_y_true
    aligned_dict['topk_predictions'] = aligned_topk
    return aligned_dict, "已按 y_true 标签序列完成稳定对齐"

def compare_baselines(pkl_path1, pkl_path2, k_list=[10, 50, 100], alpha_list=[0.01, 0.05]):
    """
    比较两个baseline的预测结果

    Args:
        pkl_path1: 第一个baseline的overall_predictions_dict.pkl路径
        pkl_path2: 第二个baseline的overall_predictions_dict.pkl路径
        k_list: 评价指标中的k值列表
        alpha_list: 显著性水平列表
    """
    print(f"读取文件1: {pkl_path1}")
    with open(pkl_path1, 'rb') as f:
        predictions_dict1 = pickle.load(f)

    print(f"读取文件2: {pkl_path2}")
    with open(pkl_path2, 'rb') as f:
        predictions_dict2 = pickle.load(f)

    for i, predictions_dict in enumerate([predictions_dict1, predictions_dict2], start=1):
        for required_key in ['y_true', 'topk_predictions']:
            if required_key not in predictions_dict:
                raise KeyError(f"文件{i}缺少关键字段 '{required_key}'，请确认传入的是 overall_predictions_dict.pkl")

    # 验证数据一致性
    assert len(predictions_dict1['y_true']) == len(predictions_dict2['y_true']), \
        f"两个文件的样本数不一致: {len(predictions_dict1['y_true'])} vs {len(predictions_dict2['y_true'])}"

    # 验证真实标签是否一致（用于配对t检验）
    y_true1 = np.array(predictions_dict1['y_true'])
    y_true2 = np.array(predictions_dict2['y_true'])
    if not np.array_equal(y_true1, y_true2):
        print("警告: 两个文件的 y_true 顺序不一致，尝试自动对齐...")
        aligned_dict2, align_msg = _try_align_by_y_true(predictions_dict1, predictions_dict2)
        print(f"  对齐结果: {align_msg}")

        if aligned_dict2 is not None:
            predictions_dict2 = aligned_dict2
            y_true2 = np.array(predictions_dict2['y_true'])

        if not np.array_equal(y_true1, y_true2):
            print("警告: 自动对齐失败，回退到逐位置过滤一致样本")
            mask = y_true1 == y_true2
            predictions_dict1['y_true'] = y_true1[mask].tolist()
            predictions_dict1['topk_predictions'] = [pred for i, pred in enumerate(predictions_dict1['topk_predictions']) if mask[i]]
            predictions_dict2['y_true'] = y_true2[mask].tolist()
            predictions_dict2['topk_predictions'] = [pred for i, pred in enumerate(predictions_dict2['topk_predictions']) if mask[i]]
            print(f"已过滤，保留 {np.sum(mask)} 个一致的样本")
        else:
            print("已完成自动对齐，将使用对齐后的顺序进行配对 t 检验")

    # 进行t检验
    results = t_test_comparison(predictions_dict1, predictions_dict2, k_list, alpha_list)

    return results


def _cascade_user_count(cas):
    """级联中用户步数（不含末尾 EOS）。与 DataConstruct.build_dataset + with_EOS 一致。"""
    if not cas:
        return 0
    if cas[-1] == Constants.EOS:
        return len(cas) - 1
    return len(cas)


def _per_sample_lengths_and_y_true_aligned_to_metrics(test_cascades, max_len):
    """
    复现 test_epoch 中 Metrics 的样本顺序：按测试集级联顺序（与 cascadetest 行序一致），
    对每条级联按时间步展开；仅保留 y_true != PAD 的位置（与 Metrics.compute_metric 一致）。
    每一步的金标签为 row[pos+1]，与 tgt[:, 1:] 展平后一致。
    展平顺序与 batch_size 无关（行优先，等价于依次遍历每条级联）。

    Returns:
        (lengths_out, y_true_out): 与 pkl 中逐条预测对齐的长度列表与真实标签列表
    """
    seq_width = max_len + 1  # 与 DataConstruct.pad_to_longest 一致
    pad = Constants.PAD
    lengths_out = []
    y_true_out = []
    for cas in test_cascades:
        user_len = _cascade_user_count(cas)
        row = cas + [pad] * (seq_width - len(cas))
        row = row[:seq_width]
        for pos in range(seq_width - 1):
            gold = row[pos + 1]
            if gold != pad:
                lengths_out.append(user_len)
                y_true_out.append(int(gold))
    return lengths_out, y_true_out


def _equal_frequency_tertile_labels(lengths):
    """
    按长度值排序后三等分样本（等频三分位），使每档样本数至多差 1。
    返回与 lengths 同长的档标签 0/1/2（0=最短档，2=最长档）。
    """
    lengths = np.asarray(lengths, dtype=np.float64)
    n = len(lengths)
    if n < 3:
        raise ValueError(f"样本数过少 (n={n})，无法划为 3 档")
    order = np.argsort(lengths, kind="mergesort")
    strata = np.empty(n, dtype=np.int32)
    chunks = np.array_split(order, 3)
    for lab, idxs in enumerate(chunks):
        strata[idxs] = lab
    return strata


def compare_stratified_performance(
    pkl_path,
    dataset_dir,
    k_list=(10, 50, 100),
    max_len=500,
):
    """
    按测试级联「用户规模」长度分层（等频三分位），统计各层样本数、Hits@k、MAP@k，
    并计算长度与逐样本指标之间的 Spearman 相关。

    Args:
        pkl_path: overall_predictions_dict.pkl，需含 topk_predictions（长度应为 max(k_list)）、y_true
        dataset_dir: 数据集根目录（含 cascadetest.txt 与 u2idx.pickle），与训练时 --data 指向的目录一致
        k_list: 与保存 pkl 时 test 所用的 k 一致；topk 预测长度应 >= max(k_list)
        max_len: 训练/测试 --max_len，须与生成 pkl 时一致（影响 padding 与截断，故必须对齐）

    Returns:
        dict: strata 分层表、spearman 各指标、分档边界说明
    """
    from utils.DataConstruct import Split_data

    print(f"读取预测: {pkl_path}")
    with open(pkl_path, "rb") as f:
        predictions_dict = pickle.load(f)
    for key in ("topk_predictions", "y_true"):
        if key not in predictions_dict:
            raise KeyError(f"缺少字段 '{key}'，请确认是 overall_predictions_dict.pkl")

    _, _, _train, _valid, test, _time_span = Split_data(
        dataset_dir, with_EOS=True, max_len=max_len
    )
    test_cascades = test[0]

    lengths, y_true_from_dataset = _per_sample_lengths_and_y_true_aligned_to_metrics(
        test_cascades, max_len=max_len
    )
    n_pred = len(predictions_dict["y_true"])
    if len(lengths) != n_pred:
        raise ValueError(
            f"由 cascadetest 展开的长度列表 ({len(lengths)}) 与 pkl 预测条数 ({n_pred}) 不一致。"
            f"请检查 dataset_dir、max_len={max_len} 是否与生成该 pkl 时一致。"
        )

    y_pkl = np.asarray(predictions_dict["y_true"], dtype=np.int64)
    y_ref = np.asarray(y_true_from_dataset, dtype=np.int64)
    if not np.array_equal(y_pkl, y_ref):
        bad = np.where(y_pkl != y_ref)[0]
        i0 = int(bad[0])
        n_show = min(5, len(bad))
        examples = [
            f"idx={int(bad[i])} pkl={int(y_pkl[bad[i]])} dataset={int(y_ref[bad[i]])}"
            for i in range(n_show)
        ]
        raise ValueError(
            "pkl 中的 y_true 与根据 cascadetest 重放的标签不一致（请检查 dataset_dir、max_len 是否与生成 pkl 时一致）。"
            f" 不一致条数: {len(bad)}/{n_pred}；前若干例: {', '.join(examples)}"
        )
    print(
        f"校验通过: y_true 与 Split_data 测试级联逐步展开一致（共 {n_pred} 条）。"
    )

    metrics_dict = compute_metrics_from_predictions(predictions_dict, k_list=k_list)
    lengths_arr = np.asarray(lengths, dtype=np.float64)
    strata = _equal_frequency_tertile_labels(lengths_arr)

    # 分档边界（各档内长度的最小/最大值，便于写论文）
    edges_desc = []
    result_strata = []
    for g in range(3):
        mask = strata == g
        n_g = int(np.sum(mask))
        lg = lengths_arr[mask]
        entry = {
            "stratum": g,
            "n": n_g,
            "length_min": float(np.min(lg)) if n_g else float("nan"),
            "length_max": float(np.max(lg)) if n_g else float("nan"),
        }
        for k in k_list:
            hits = np.asarray(metrics_dict[f"hits@{k}"])[mask]
            maps = np.asarray(metrics_dict[f"map@{k}"])[mask]
            entry[f"mean_hits@{k}"] = float(np.mean(hits)) if n_g else float("nan")
            entry[f"mean_map@{k}"] = float(np.mean(maps)) if n_g else float("nan")
        result_strata.append(entry)
        edges_desc.append((entry["length_min"], entry["length_max"]))

    spearman_out = {}
    for k in k_list:
        hits = np.asarray(metrics_dict[f"hits@{k}"])
        maps = np.asarray(metrics_dict[f"map@{k}"])
        rho_h, p_h = stats.spearmanr(lengths_arr, hits)
        rho_m, p_m = stats.spearmanr(lengths_arr, maps)
        spearman_out[f"hits@{k}"] = {
            "rho": float(rho_h) if not np.isnan(rho_h) else None,
            "p_value": float(p_h) if not np.isnan(p_h) else None,
        }
        spearman_out[f"map@{k}"] = {
            "rho": float(rho_m) if not np.isnan(rho_m) else None,
            "p_value": float(p_m) if not np.isnan(p_m) else None,
        }

    print("\n=== Stratified performance (equal-frequency tertiles by cascade user count) ===")
    print(
        "长度定义: 每条测试级联的用户数（含截断至 max_len 后的用户数，不含 EOS）。"
        "同一级联内各预测步共享该长度。"
    )
    for row in result_strata:
        print(
            f"\nStratum {row['stratum']} | n={row['n']} | "
            f"length in [{row['length_min']:.1f}, {row['length_max']:.1f}]"
        )
        for k in k_list:
            print(
                f"  hits@{k}: {row[f'mean_hits@{k}']:.6f} | "
                f"map@{k}: {row[f'mean_map@{k}']:.6f}"
            )

    print("\n=== Spearman (cascade user count vs per-sample metric) ===")
    for k in k_list:
        h = spearman_out[f"hits@{k}"]
        m = spearman_out[f"map@{k}"]
        print(
            f"hits@{k}: rho={h['rho']}, p={h['p_value']}\n"
            f"map@{k}: rho={m['rho']}, p={m['p_value']}"
        )

    return {
        "strata": result_strata,
        "spearman": spearman_out,
        "length_definition": "per-cascade user count (excluding EOS), shared by all steps",
        "stratification": "equal-frequency tertiles on lengths (sorted sample thirds)",
        "stratum_length_ranges": edges_desc,
    }

if __name__ == '__main__':
    # 原有的analyze_diff功能
    # result_folder = 'Information_Diffusion/DisenIDP/checkpoint/Douban/06_05/20_50_38__epoch-200_batch-16'
    # result_path = os.path.join(result_folder, 'test.csv')
    # result2_path = os.path.join(result_folder, 'test2.csv')
    # analyze_diff(result_path, result2_path)

    # 新增：比较两个baseline的预测结果
    # 示例用法：
    '''pkl_path1 = './checkpoint/Twitter_0607/26-04-04/14-46-09_10/overall_predictions_dict.pkl'
    pkl_path2 = '../RotDiff/checkpoint/Twitter_0607/26-04-04/12-26-56_10/overall_predictions_dict.pkl'

    # 如果文件存在，进行比较
    if os.path.exists(pkl_path1) and os.path.exists(pkl_path2):
        compare_baselines(pkl_path1, pkl_path2, k_list=[10, 50, 100], alpha_list=[0.01, 0.05])
    else:
        print("请设置正确的pkl文件路径")'''
    dataset = 'KuaiRand'
    pkl_path = f'./checkpoint/{dataset}/26-04-05/00-16-37_10/overall_predictions_dict.pkl'
    dataset_dir = f'../data/{dataset}'

    compare_stratified_performance(pkl_path, dataset_dir, k_list=[10, 50, 100], max_len=500)