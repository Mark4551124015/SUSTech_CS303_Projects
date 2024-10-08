import logging
import time
from copy import deepcopy
from sklearn.model_selection import KFold

import math
import numpy as np
import torch
from pytorch_lightning import seed_everything
from sklearn.metrics import roc_auc_score
import os
from demo_final.kgrs import KGRS


def nDCG(sorted_items, pos_item, train_pos_item, k=5):
    dcg = 0
    train_pos_item = set(train_pos_item)
    filter_item = set(filter(lambda item: item not in train_pos_item, pos_item))
    max_correct = min(len(filter_item), k)
    train_hit_num = 0
    valid_num = 0
    recommended_items = set()
    for index in range(len(sorted_items)):
        if sorted_items[index] in train_pos_item:
            train_hit_num += 1
        else:
            valid_num += 1
            if sorted_items[index] in filter_item and sorted_items[index] not in recommended_items:
                dcg += 1 / math.log2(index - train_hit_num + 2)  # rank从0开始算
                recommended_items.add(sorted_items[index])
            if valid_num >= k:
                break
    idcg = sum([1 / math.log2(i + 2) for i in range(max_correct)])
    return dcg / idcg

import random
def load_data():
    train_pos, train_neg = np.load("./data/train_pos.npy"), np.load("./data/train_neg.npy")
    kf = KFold(4)

    train_pos_5_g = kf.split(train_pos)
    train_neg_5_g = kf.split(train_neg)

    for i, ((tra_p_i, val_p_i), (tra_n_i, val_n_i)) in enumerate(zip(train_pos_5_g, train_neg_5_g)):
        if i != 3 : continue
        tra_p, val_p, tra_n, val_n = [], [], [], []
        for ind, i in enumerate(tra_p_i):
            if i > 513: break
            tra_p.append(train_pos[i])
        for ind,i in enumerate(val_p_i):
            if i > 513: break
            val_p.append(train_pos[i])
        for ind,i in enumerate(tra_n_i):
            if i > 513: break
            tra_n.append(train_neg[i])
        for ind,i in enumerate(val_n_i):
            if ind > 513: break
            val_n.append(train_neg[i])
        yield np.array(tra_p),  np.array(tra_n),  np.array(val_p),  np.array(val_n)


def get_user_pos_items(train_pos, test_pos):
    user_pos_items, user_train_pos_items = {}, {}
    for record in train_pos:
        user, item = record[0], record[1]
        if user not in user_train_pos_items:
            user_train_pos_items[user] = set()
        user_train_pos_items[user].add(item)
    for record in test_pos:
        user, item = record[0], record[1]
        if user not in user_train_pos_items:
            user_train_pos_items[user] = set()
        if user not in user_pos_items:
            user_pos_items[user] = set()
        user_pos_items[user].add(item)
    return user_pos_items, user_train_pos_items


def evaluate():
    data_gen = load_data()
    for train_pos, train_neg, test_pos, test_neg in data_gen:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        user_pos_items, user_train_pos_items = get_user_pos_items(train_pos=train_pos, test_pos=test_pos)
        logging.disable(logging.INFO)
        seed_everything(1088, workers=True)
        torch.set_num_threads(8)
        auc, ndcg5 = 0, 0
        init_timeout, train_timeout, ctr_timeout, topk_timeout = False, False, False, False
        start_time, init_time, train_time, ctr_time, topk_time = time.time(), 0, 0, 0, 0
        kgrs = KGRS(train_pos=deepcopy(train_pos),
                    train_neg=deepcopy(train_neg),
                    kg_lines=open('./data/kg.txt', encoding='utf-8').readlines())
        init_time = time.time() - start_time

        kgrs.training()
        train_time = time.time() - start_time - init_time

        test_data = np.concatenate((deepcopy(test_neg), deepcopy(test_pos)), axis=0)
        np.random.shuffle(test_data)
        print(test_data)
        test_label = test_data[:, 2]
        test_data = test_data[:, :2]
        print(len(test_label), len(test_data))

        kgrs.eval_ctr(test_data)
        scores = kgrs.eval_ctr(test_data=test_data)
        auc = roc_auc_score(y_true=test_label, y_score=scores)
        ctr_time = time.time() - start_time - init_time - train_time

        users = list(user_pos_items.keys())
        user_item_lists = kgrs.eval_topk(users=users)
        ndcg5 = np.mean([nDCG(user_item_lists[index], user_pos_items[user], user_train_pos_items[user]) for index, user in
                        enumerate(users)])

        topk_time = time.time() - start_time - init_time - train_time - ctr_time
        print( auc, ndcg5, init_timeout, train_timeout, ctr_timeout, topk_timeout, init_time, train_time, ctr_time, topk_time)



if __name__ == '__main__':
    start = time.time()
    evaluate()
    print(time.time() - start)
