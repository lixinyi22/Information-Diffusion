import os
import argparse

root_path = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(root_path)
data_path = os.path.join(repo_root, "data") + os.sep

parser = argparse.ArgumentParser()
# 本地数据集参数
parser.add_argument("--data", type=str, default="KuaiRand",
                        help='dataset name')
parser.add_argument('--max_len', type=int, default=500,
                        help='The max length for the model, As information is a sequential task, we should strictly control the max length to be predicted.')
# 模型控制参数
parser.add_argument('-d_model', type=int, default=64)
parser.add_argument('-time_step_split', type=int, default=5000)
parser.add_argument('-num_seq_layers', type=int, default=1)
parser.add_argument('-num_mm_layers', type=int, default=2)
parser.add_argument('-num_hg_layers', type=int, default=2)
parser.add_argument('-num_heads', type=int, default=8)
parser.add_argument('-knn_k', type=int, default=10)
parser.add_argument(
    '--ablation_mode',
    type=str,
    default='wo_structural',
    choices=[
        'full',
        'wo_structural', 'wo_interactive', 'wo_contextual',
        'only_structural', 'only_interactive', 'only_contextual',
    ],
    help='Ablation: full; wo_* = drop one channel; only_* = single-channel proxy (IV diagnostics).'
)
parser.add_argument(
    '--fusion_mode',
    type=str,
    default='sum',
    choices=['sum', 'sum_uc', 'concat', 'concat_uc'],
    help='Fusion variants: sum (default), sum_uc, concat, concat_uc.'
)
parser.add_argument(
    '--fusion_weight_mode',
    type=str,
    default='disable',
    choices=['manual', 'adaptive', 'disable'],
    help='manual: fixed fusion_w_*; adaptive: learn softmax weights (wg+wi+wc=3); disable: fixed wg=wi=wc=1.',
)
parser.add_argument(
    '--fusion_w_g',
    type=float,
    default=1.0,
    help='Manual fusion weight for structural (g); also used as init proportion for adaptive (log).',
)
parser.add_argument(
    '--fusion_w_i',
    type=float,
    default=1.0,
    help='Manual fusion weight for interactive (i); also adaptive init proportion.',
)
parser.add_argument(
    '--fusion_w_c',
    type=float,
    default=1.0,
    help='Manual fusion weight for contextual (c); also adaptive init proportion.',
)
parser.add_argument(
    '--disable_self_gate',
    action='store_true',
    help='Ablation switch: disable self-gating module.'
)
parser.add_argument(
    '--use_history_user_content',
    action='store_true',
    help='Use user content embedding aggregated from user-history info interactions.'
)
# 训练参数
parser.add_argument('-epoch', type=int, default=200)
parser.add_argument('-batch_size', type=int, default=32)
parser.add_argument('-dropout', type=float, default=0.1)
parser.add_argument('-lr', type=float, default=1e-3)
parser.add_argument('-l2', type=float, default=0)
parser.add_argument('-warmup', type=int, default=10)
parser.add_argument('-n_warmup_steps', type=int, default=1000)
parser.add_argument('-patience', type=int, default=10, help="control the step of early-stopping")
parser.add_argument(
    '--profile_stop_epoch',
    type=int,
    default=0,
    help='Stop after this many completed epochs, log [Profile] full_summary (with avg timings), then exit without test. 0=disabled (full train+test).',
)
parser.add_argument('-seed', type=int, default=2026)
# 日志及保存参数
parser.add_argument('-save_path', default=os.path.join(root_path, "checkpoint"))
parser.add_argument('--gpu', type=str, default="0", help="GPU id")
opt = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import pickle

import time
from time import perf_counter
import random
import numpy as np
import torch
import torch.nn as nn
import datetime
import gc
from tqdm import tqdm
import logging
import json
from typing import Any, Dict, List, Optional

from utils.DataConstruct import DataConstruct, Split_data, LoadContentEmbedding
from utils.GraphConstruct import LoadDiffusionGraph, BuildUserInfoParticipation
from utils.Optim import ScheduledOptim
from utils.Metrics import Metrics
from model.model import MyModel
import utils.Constants as Constants
from utils.compute_stats import log_parameter_and_flops


def build_profile_summary(
    profile_payload: Dict[str, Any],
    epoch_timings: List[Dict[str, Any]],
    test_inference_time_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """Merge params/GFLOPs with per-epoch timings and arithmetic means of train / val-infer times."""
    n = len(epoch_timings)
    if n > 0:
        avg_train = sum(e["train_time_sec"] for e in epoch_timings) / n
        avg_val = sum(e["validation_inference_time_sec"] for e in epoch_timings) / n
    else:
        avg_train = 0.0
        avg_val = 0.0
    out = {
        **profile_payload,
        "epoch_timings": epoch_timings,
        "avg_train_time_sec": float(avg_train),
        "avg_validation_inference_time_sec": float(avg_val),
        "num_epochs_recorded": int(n),
    }
    if test_inference_time_sec is not None:
        out["test_inference_time_sec"] = float(test_inference_time_sec)
    return out


def get_performance(crit, pred, gold):
    loss = crit(pred, gold.contiguous().view(-1))
    pred = pred.max(1)[1]

    gold = gold.contiguous().view(-1)
    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum().float()
    return loss, n_correct

def train_epoch(model, training_data, diffusion_graph, loss_func, optimizer):
    ''' Epoch operation in training phase'''
    model.train()

    total_loss = 0.0
    total_task_loss = 0.0
    total_contrastive_loss = 0.0
    n_total_words = 0.0
    n_total_correct = 0.0

    for i, batch in enumerate(tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False, ncols=100)):
        # prepare data
        tgt, tgt_timestamp, tgt_content_embedding, tgt_id = (item.cuda() for item in batch)

        gold = tgt[:, 1:]

        n_words = gold.data.ne(Constants.PAD).sum().float()
        n_total_words += n_words

        optimizer.zero_grad()
        pred, contrastive_loss = model(tgt, tgt_timestamp, tgt_id, diffusion_graph)

        # backward
        task_loss, n_correct = get_performance(loss_func, pred, gold)
        # 总损失 = 任务损失 + 对比学习损失
        #weighted_contrastive_loss = model.contrastive_weight * contrastive_loss
        loss = task_loss
        #loss = task_loss
        loss.backward()

        # update parameters
        optimizer.step()
        optimizer.update_learning_rate()

        # note keeping
        n_total_correct += n_correct
        total_loss += loss.item()
        total_task_loss += task_loss.item()
        total_contrastive_loss += 0#weighted_contrastive_loss.item()

    return total_loss/n_total_words, n_total_correct/n_total_words, total_task_loss/n_total_words, total_contrastive_loss/n_total_words

def test_epoch(model, validation_data, diffusion_graph, k_list=[10, 50, 100]):
    ''' Epoch operation in evaluation phase '''
    model.eval()

    metric = Metrics()
    scores = {}
    for k in k_list:
        scores['hits@' + str(k)] = 0
        scores['map@' + str(k)] = 0
    overall_embedding_dict = {
        'user_graph_emb': None,
        'user_inter_emb': None,
        'info_graph_emb': None,
        'info_inter_emb': None,
        'info_content_emb': None
    }
    overall_predictions_dict = {
        'topk_predictions': [],
        'y_true': []
    }
    # activation stats for 6 interaction terms (heatmap data)
    activation_sum = {
        'vs_us': 0.0,
        'vi_ui': 0.0,
        'vs_ui': 0.0,
        'vi_us': 0.0,
        'vc_us': 0.0,
        'vc_ui': 0.0,
    }
    activation_cnt = 0

    n_total_words = 0
    y_pred, y_gold = None, None
    with torch.no_grad():
        for i, batch in enumerate(tqdm(validation_data, mininterval=2, desc='  - (Validation) ', leave=False, ncols=100)):
            #print("Validation batch ", i)
            # prepare data
            tgt, tgt_timestamp, tgt_content_embedding, tgt_id = (item.cuda() for item in batch)
            y_gold = tgt[:, 1:].contiguous().view(-1).detach().cpu().numpy()

            # forward
            pred, embedding_dict = model.inference(tgt, tgt_timestamp, tgt_content_embedding, diffusion_graph)
            #pred = model.inference(tgt, tgt_timestamp, tgt_id, diffusion_graph)
            y_pred = pred.detach().cpu().numpy()

            scores_batch, scores_len, predictions_dict = metric.compute_metric(y_pred, y_gold, k_list)
            n_total_words += scores_len
            for k in k_list:
                scores['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
                scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len

            # 收集预测结果
            overall_predictions_dict['topk_predictions'].extend(predictions_dict['topk_predictions'])
            overall_predictions_dict['y_true'].extend(predictions_dict['y_true'])

            # 对每个embedding进行堆叠处理
            for key in overall_embedding_dict.keys():
                # 与 info_graph / info_inter 行数一致：按有效位置的内容向量（非 batch 粒度的 info_content_emb）
                if key == "info_content_emb":
                    emb = embedding_dict["info_content_seq"]
                else:
                    emb = embedding_dict[key]
                embedding_np = emb.detach().cpu().numpy()
                if key in ('user_graph_emb', 'user_inter_emb'):
                    overall_embedding_dict[key] = embedding_np
                    continue
                if overall_embedding_dict[key] is not None:
                    overall_embedding_dict[key] = np.concatenate([overall_embedding_dict[key], embedding_np], axis=0)
                else:
                    overall_embedding_dict[key] = embedding_np

            # ===== activation heatmap stats (align each position to gold user) =====
            # IMPORTANT: model.inference() uses input = tgt[:, :-1] and returns per-position embeddings
            # masked by valid positions of that input. To keep shapes consistent, we must use the same
            # validity mask here, then index gold users from tgt[:, 1:] with that mask.
            x_in_t = tgt[:, :-1].contiguous().view(-1)   # on GPU
            y_gold_t = tgt[:, 1:].contiguous().view(-1) # on GPU
            valid_pos = (x_in_t != Constants.PAD)
            if valid_pos.any():
                gold_users = y_gold_t[valid_pos].long()
                # per-position info embeddings already masked to valid positions
                v_s = embedding_dict['info_graph_emb']          # (n_valid, d)
                v_i = embedding_dict['info_inter_emb']          # (n_valid, d)
                v_c = embedding_dict.get('info_content_seq')    # (n_valid, d)
                u_s_all = embedding_dict['user_graph_emb']      # (user_size, d)
                u_i_all = embedding_dict['user_inter_emb']      # (user_size, d)
                u_s = u_s_all[gold_users]                       # (n_valid, d)
                u_i = u_i_all[gold_users]                       # (n_valid, d)

                # dot products per position (scalar)
                t_vs_us = torch.sum(v_s * u_s, dim=-1)
                t_vi_ui = torch.sum(v_i * u_i, dim=-1)
                t_vs_ui = torch.sum(v_s * u_i, dim=-1)
                t_vi_us = torch.sum(v_i * u_s, dim=-1)
                if v_c is not None:
                    t_vc_us = torch.sum(v_c * u_s, dim=-1)
                    t_vc_ui = torch.sum(v_c * u_i, dim=-1)
                else:
                    t_vc_us = torch.zeros_like(t_vs_us)
                    t_vc_ui = torch.zeros_like(t_vs_us)

                activation_sum['vs_us'] += torch.sum(torch.abs(t_vs_us)).item()
                activation_sum['vi_ui'] += torch.sum(torch.abs(t_vi_ui)).item()
                activation_sum['vs_ui'] += torch.sum(torch.abs(t_vs_ui)).item()
                activation_sum['vi_us'] += torch.sum(torch.abs(t_vi_us)).item()
                activation_sum['vc_us'] += torch.sum(torch.abs(t_vc_us)).item()
                activation_sum['vc_ui'] += torch.sum(torch.abs(t_vc_ui)).item()
                activation_cnt += int(t_vs_us.numel())

        for k in k_list:
            scores['hits@' + str(k)] = scores['hits@' + str(k)] / n_total_words
            scores['map@' + str(k)] = scores['map@' + str(k)] / n_total_words

    activation_stats = None
    if activation_cnt > 0:
        activation_stats = {k: (v / activation_cnt) for k, v in activation_sum.items()}
        activation_stats['count'] = activation_cnt


    return scores, overall_embedding_dict, overall_predictions_dict, activation_stats

def train_model(data_path):
    # ========= Preparing DataLoader =========#
    user_size, info_size, train, valid, test, time_interval = Split_data(data_path, with_EOS=True, max_len=opt.max_len)

    train_data = DataConstruct(train, batch_size=opt.batch_size, cuda=False, shuffle=True, test=False, with_EOS=True, max_len=opt.max_len)
    valid_data = DataConstruct(valid, batch_size=opt.batch_size, cuda=False, shuffle=False, test=False, with_EOS=True, max_len=opt.max_len)
    test_data = DataConstruct(test, batch_size=opt.batch_size, cuda=False, shuffle=False, test=False, with_EOS=True, max_len=opt.max_len)

    opt.user_size = user_size
    opt.info_size = info_size
    opt.time_interval = time_interval

    diffusion_graph = LoadDiffusionGraph(train[0], user_size)
    user_info_participation = BuildUserInfoParticipation(train[0], user_size, info_size)
    content_embedding = LoadContentEmbedding(data_path)

    # ========= Preparing Model =========#
    model = MyModel(opt, content_embedding)

    loss_func = nn.CrossEntropyLoss(reduction='sum', ignore_index=Constants.PAD) # size_average=False,

    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizerAdam = torch.optim.Adam(params, lr=opt.lr, betas=(0.9, 0.98),weight_decay = opt.l2, eps=1e-09)  # weight_decay is l2 regularization
    optimizer = ScheduledOptim(optimizerAdam, opt.d_model, opt.n_warmup_steps)

    if torch.cuda.is_available():
        model = model.cuda()
        loss_func = loss_func.cuda()

    content_emb_dim = int(content_embedding.shape[1])
    profile_payload = log_parameter_and_flops(
        model, opt, diffusion_graph, content_emb_dim
    )
    epoch_timings = []
    if opt.profile_stop_epoch and opt.profile_stop_epoch > opt.epoch:
        logging.warning(
            "[Profile] -epoch (%s) < profile_stop_epoch (%s); profiling early stop will not trigger.",
            opt.epoch,
            opt.profile_stop_epoch,
        )

    validation_history = 0.0
    patience_counter = 0
    best_scores = {}
    for epoch_i in range(opt.epoch):
        logging.info(f'\n[ Epoch {epoch_i}]')

        gc.collect()
        torch.cuda.empty_cache()
        t0 = perf_counter()
        train_loss, train_accu, train_task_loss, train_contrastive_loss = train_epoch(model, train_data, diffusion_graph, loss_func, optimizer)
        train_sec = perf_counter() - t0
        task_ratio = train_task_loss / train_loss if train_loss > 0 else 0
        contrastive_ratio = train_contrastive_loss / train_loss if train_loss > 0 else 0
        logging.info(
            f'  - (Training)   total loss: {train_loss}, task_loss: {train_task_loss} ({task_ratio:.4f}), contrastive_loss: {train_contrastive_loss} ({contrastive_ratio:.4f}), accuracy: {100 * train_accu} %, elapse: {(train_sec / 60)} min'
        )

        t1 = perf_counter()
        scores, _, _, _ = test_epoch(model, valid_data, diffusion_graph)
        val_infer_sec = perf_counter() - t1
        epoch_timings.append(
            {
                "epoch": int(epoch_i),
                "train_time_sec": float(train_sec),
                "validation_inference_time_sec": float(val_infer_sec),
            }
        )
        logging.info(
            f"[Profile] epoch={epoch_i} train_time_sec={train_sec:.4f} validation_inference_time_sec={val_infer_sec:.4f}"
        )
        logging.info('  - ( Validation )) ')
        for metric in scores.keys():
            logging.info(metric + ' ' + str(scores[metric]))
        logging.info(f"Validation use time: {val_infer_sec / 60} min")

        if opt.profile_stop_epoch and (epoch_i + 1) >= opt.profile_stop_epoch:
            profile_summary = build_profile_summary(profile_payload, epoch_timings, None)
            logging.info(
                "[Profile] full_summary=%s",
                json.dumps(profile_summary, ensure_ascii=False),
            )
            logging.info(
                "[Profile] Stopping for profiling after %s epoch(s) (profile_stop_epoch=%s).",
                epoch_i + 1,
                opt.profile_stop_epoch,
            )
            return

        if validation_history <= scores["hits@100"]:
            logging.info(f"Best Validation hit@100:{scores['hits@100']} at Epoch:{epoch_i}")
            validation_history = scores["hits@100"]
            best_scores = {
                'epoch': epoch_i,
                'validation': scores
            }
            logging.info("Save best model!!!")
            torch.save(model.state_dict(), opt.save_path)
            patience_counter = 0
        else:
            logging.info(f"No improvement. Best validation hit@100: {validation_history} at Epoch:{best_scores['epoch']}")
            patience_counter += 1
        if patience_counter >= opt.patience:
            logging.info("Early_Stopping")
            break

    model.load_state_dict(torch.load(opt.save_path, map_location='cuda:0'))
    logging.info('  - (Test) ')
    t_test = perf_counter()
    scores, _, _, _ = test_epoch(model, test_data, diffusion_graph)
    test_infer_sec = perf_counter() - t_test
    logging.info(f"Test use time: {test_infer_sec / 60} min")
    logging.info(f"[Profile] test_inference_time_sec={test_infer_sec:.4f}")

    profile_summary = build_profile_summary(
        profile_payload, epoch_timings, float(test_infer_sec)
    )
    logging.info("[Profile] full_summary=%s", json.dumps(profile_summary, ensure_ascii=False))
    logging.info("Corresponding test scores:")
    for metric in scores.keys():
        logging.info(metric + ' ' + str(scores[metric]))

def test_model(data_path):
    # 修复：与 train_model 保持一致，使用 with_EOS=True
    user_size, info_size, train, valid, test, time_interval = Split_data(data_path, with_EOS=True, max_len=opt.max_len)

    test_data = DataConstruct(test, batch_size=opt.batch_size, cuda=False, shuffle=False, test=False, with_EOS=True, max_len=opt.max_len)
    opt.user_size = user_size
    opt.info_size = info_size
    opt.time_interval = time_interval
    diffusion_graph = LoadDiffusionGraph(train[0], user_size)
    user_info_participation = BuildUserInfoParticipation(train[0], user_size, info_size)
    content_embedding = LoadContentEmbedding(data_path)

    model = MyModel(opt, content_embedding, user_info_participation=user_info_participation)
    model.load_state_dict(torch.load(opt.save_path, map_location='cuda:0'))
    if torch.cuda.is_available():
        model = model.cuda()

    # 确保模型处于评估模式
    model.eval()

    logging.info('  - (Test) ')
    start_test = time.time()
    # save activation stats for heatmap
    scores, overall_embedding_dict, overall_predictions_dict, activation_stats = test_epoch(model, test_data, diffusion_graph)
    logging.info(f"Test use time: {(time.time() - start_test) / 60} min")
    logging.info("Corresponding test scores:")
    for metric in scores.keys():
        logging.info(metric + ' ' + str(scores[metric]))
    logging.info(f"Test Finished")

    '''with open(os.path.join(opt.save_dir, f"activation_stats.pkl"), "wb") as f:
        pickle.dump(activation_stats, f)
    with open(os.path.join(opt.save_dir, f"overall_embedding_dict.pkl"), "wb") as f:
        pickle.dump(overall_embedding_dict, f)'''
    with open(os.path.join(opt.save_dir, f"overall_predictions_dict.pkl"), "wb") as f:
        pickle.dump(overall_predictions_dict, f)



if __name__ == "__main__":
    # 参数处理
    opt.data_path = data_path + opt.data

    dt = datetime.datetime.now()
    date, t = dt.strftime("%y-%m-%d"), dt.strftime('%H-%M-%S_')
    f_name = t + str(opt.patience)
    opt.save_dir = os.path.join(opt.save_path, opt.data, date, f_name)
    os.makedirs(opt.save_dir, exist_ok=True)
    opt.save_path = os.path.join(opt.save_dir, f"best.pt")

    handlers = [logging.StreamHandler()]
    filename = os.path.join(opt.save_dir, f'log.log')
    handlers.append(logging.FileHandler(filename, mode='w'))  # 添加mode='w'确保文件被正确打开

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True  # 强制重新配置logging
    )

    # 打印参数
    logging.info("*" * 30)
    in_opt_dict = vars(opt)
    for key in in_opt_dict.keys():
        logging.info(f"{key}={in_opt_dict[key]}")
    logging.info("*" * 30)

    # 设置随机种子
    logging.info(f"Setting random seed to {opt.seed}")
    random.seed(opt.seed)
    os.environ['PYTHONHASHSEED'] = str(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.cuda.empty_cache()
    np.random.seed(opt.seed)
    np.set_printoptions(threshold=np.inf)

    #train_model(opt.data_path)
    # test the model
    opt.save_dir = os.path.join(root_path, "checkpoint", opt.data, "26-04-07", "19-38-56_10")
    opt.save_path = os.path.join(opt.save_dir, f"best.pt")
    test_model(opt.data_path)

    for handler in logging.getLogger().handlers:
        handler.flush()


