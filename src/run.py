import os
import pickle
import argparse
import time
import random
import numpy as np 
import torch
import torch.nn as nn
import datetime
import gc
from tqdm import tqdm
import logging

from utils.DataConstruct import DataConstruct, Split_data, LoadContentEmbedding
from utils.GraphConstruct import LoadDiffusionGraph
from utils.Optim import ScheduledOptim
from utils.Metrics import Metrics
from model.model import MyModel
import utils.Constants as Constants

root_path = './' 
data_path = "./data/"

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
        weighted_contrastive_loss = model.contrastive_weight * contrastive_loss
        loss = task_loss + weighted_contrastive_loss
        #loss = task_loss
        loss.backward()

        # update parameters
        optimizer.step()
        optimizer.update_learning_rate()

        # note keeping
        n_total_correct += n_correct
        total_loss += loss.item()
        total_task_loss += task_loss.item()
        total_contrastive_loss += weighted_contrastive_loss.item()

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

    n_total_words = 0
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
                embedding_np = embedding_dict[key].detach().cpu().numpy()
                if key == 'user_graph_emb' or key == 'user_inter_emb':
                    overall_embedding_dict[key] = embedding_np
                    continue
                if overall_embedding_dict[key] is not None:
                    overall_embedding_dict[key] = np.concatenate([overall_embedding_dict[key], embedding_np], axis=0)
                else:
                    overall_embedding_dict[key] = embedding_np

        for k in k_list:
            scores['hits@' + str(k)] = scores['hits@' + str(k)] / n_total_words
            scores['map@' + str(k)] = scores['map@' + str(k)] / n_total_words

    return scores, overall_embedding_dict, overall_predictions_dict

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
    content_embedding = LoadContentEmbedding(data_path)

    # ========= Preparing Model =========#
    model = MyModel(opt, content_embedding)
    logging.info(f"The model have {sum(x.numel() for x in model.parameters())} paramerters in total")
    
    loss_func = nn.CrossEntropyLoss(size_average=False, ignore_index=Constants.PAD)
    
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizerAdam = torch.optim.Adam(params, lr=opt.lr, betas=(0.9, 0.98),weight_decay = opt.l2, eps=1e-09)  # weight_decay is l2 regularization
    optimizer = ScheduledOptim(optimizerAdam, opt.d_model, opt.n_warmup_steps)

    if torch.cuda.is_available():
        model = model.cuda()
        loss_func = loss_func.cuda()

    validation_history = 0.0
    patience_counter = 0
    best_scores = {}
    for epoch_i in range(opt.epoch):
        logging.info(f'\n[ Epoch {epoch_i}]')

        gc.collect()
        torch.cuda.empty_cache()
        start = time.time()
        train_loss, train_accu, train_task_loss, train_contrastive_loss = train_epoch(model, train_data, diffusion_graph, loss_func, optimizer)
        task_ratio = train_task_loss / train_loss if train_loss > 0 else 0
        contrastive_ratio = train_contrastive_loss / train_loss if train_loss > 0 else 0
        logging.info(f'  - (Training)   total loss: {train_loss}, task_loss: {train_task_loss} ({task_ratio:.4f}), contrastive_loss: {train_contrastive_loss} ({contrastive_ratio:.4f}), accuracy: {100 * train_accu} %, elapse: {((time.time() - start) / 60)} min')

        start = time.time()
        scores, _, _ = test_epoch(model, valid_data, diffusion_graph)
        logging.info('  - ( Validation )) ')
        for metric in scores.keys():
            logging.info(metric + ' ' + str(scores[metric]))
        logging.info(f"Validation use time: {(time.time() - start) / 60} min")

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
    start_test = time.time()
    scores, _, _ = test_epoch(model, test_data, diffusion_graph)
    logging.info(f"Test use time: {(time.time() - start_test) / 60} min")
    logging.info("Corresponding test scores:")
    for metric in scores.keys():
        logging.info(metric + ' ' + str(scores[metric]))
    
def test_model(data_path):
    user_size, info_size, train, valid, test, time_interval = Split_data(data_path, with_EOS=True, max_len=opt.max_len)

    test_data = DataConstruct(test, batch_size=opt.batch_size, cuda=False, shuffle=False, test=False, with_EOS=True, max_len=opt.max_len)
    opt.user_size = user_size
    opt.info_size = info_size
    opt.time_interval = time_interval
    diffusion_graph = LoadDiffusionGraph(train[0], user_size)
    content_embedding = LoadContentEmbedding(data_path)
    
    model = MyModel(opt, content_embedding)
    model.load_state_dict(torch.load(opt.save_path, map_location='cuda:0'))
    if torch.cuda.is_available():
        model = model.cuda()

    logging.info('  - (Test) ')
    start_test = time.time()
    scores, overall_embedding_dict, overall_predictions_dict = test_epoch(model, test_data, diffusion_graph)
    logging.info(f"Test use time: {(time.time() - start_test) / 60} min")
    logging.info("Corresponding test scores:")
    for metric in scores.keys():
        logging.info(metric + ' ' + str(scores[metric]))
    logging.info(f"Test Finished")
    
    # 保存embedding和预测结果
    
    with open(os.path.join(opt.save_dir, f"overall_embedding_dict.pkl"), "wb") as f:
        pickle.dump(overall_embedding_dict, f)
    with open(os.path.join(opt.save_dir, f"overall_predictions_dict.pkl"), "wb") as f:
        pickle.dump(overall_predictions_dict, f)

parser = argparse.ArgumentParser()
# 本地数据集参数
parser.add_argument("--data", type=str, default="Twitter", 
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
parser.add_argument('-knn_k', type=int, default=100)
parser.add_argument('-contrastive_weight', type=float, default=0) 
parser.add_argument('-contrastive_temperature', type=float, default=0.9)
# 训练参数
parser.add_argument('-epoch', type=int, default=200) 
parser.add_argument('-batch_size', type=int, default=32)
parser.add_argument('-dropout', type=float, default=0.1)  
parser.add_argument('-lr', type=float, default=1e-3)
parser.add_argument('-l2', type=float, default=0)
parser.add_argument('-warmup', type=int, default=10)
parser.add_argument('-n_warmup_steps', type=int, default=1000)
parser.add_argument('-patience', type=int, default=10, help="control the step of early-stopping")
parser.add_argument('-seed', type=int, default=200)
# 日志及保存参数
parser.add_argument('-save_path', default=os.path.join(root_path, "checkpoint"))
opt = parser.parse_args() 

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
    
    train_model(opt.data_path)
    # test the model
    #opt.save_dir = 'your checkpoint path'
    #opt.save_path = os.path.join(opt.save_dir, f"best.pt")
    #test_model(opt.data_path)
    
    for handler in logging.getLogger().handlers:
        handler.flush()


