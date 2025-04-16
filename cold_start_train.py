# -*- coding: utf-8 -*-
"""
Created based on TensorFlow implementation
Converted to PyTorch with PPO reinforcement learning
"""
import logging
import random
import sys
import os
import torch
import numpy as np
import time
import heapq
import pickle
from tqdm import tqdm

# Import the PPO model instead of original InteractiveModel
from PPOInteractive_Model import get_cum_interesting, get_initial_masking, get_masking, PPOInteractiveModel

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def inter_diversity(lists):
    """Compute diversity among recommendation lists"""
    user_nb = len(lists)
    sumH = 0.0
    print(user_nb)
    for i in range(user_nb):
        for j in range(user_nb):
            if i == j:
                continue

            H_ij = 1.0 - float(len(set(lists[i]) & set(lists[j]))) / len(lists[i])
            sumH += H_ij

    Hu = sumH / (user_nb * (user_nb - 1))
    return Hu


def read_data(file):
    """Read data from dataset file and process it"""
    f = open(file, 'r')
    users = {}
    user_dict = {}
    item_dict = {}

    user_count = 0
    item_count = 1
    for line in f:
        data = line.split('::')
        user = data[0]
        item = data[1]
        rating = data[2]
        time_stmp = data[3][:-1]
        if int(user) not in user_dict:
            user_dict[int(user)] = user_count
            user_count += 1
        if int(item) not in item_dict:
            item_dict[int(item)] = item_count
            item_count += 1

        user = user_dict[int(user)]
        item = item_dict[int(item)]

        if int(user) not in users:
            users[int(user)] = []

        users[int(user)].append((int(user), int(item), float(rating), int(time_stmp)))

    f.close()
    new_users = {}
    user_historicals = {}

    for user in users:
        new_users[user] = sorted(users[user], key=lambda a: a[-1])
        user_historicals[user] = [d[1] for d in new_users[user]]

    return user_historicals, user_count, item_count


def gen_1m_train_test(path, fold, filter_nb=0):
    """Generate train-test split for MovieLens 1M dataset"""
    user_historicals, user_count, item_count = read_data(path)

    ftr = open(f'1m_train_user_{fold}', 'wb')
    fte = open(f'1m_test_user_{fold}', 'wb')
    train_users = []
    test_users = []
    for user in user_historicals:
        if filter_nb > 0 and len(user_historicals[user]) < filter_nb:
            continue
        if random.random() < 0.9:
            train_users.append(user)
        else:
            test_users.append(user)

    pickle.dump(train_users, ftr)
    ftr.close()
    pickle.dump(test_users, fte)
    fte.close()


def gen_100k_train_test(path, fold):
    """Generate train-test split for MovieLens 100K dataset"""
    user_historicals, user_count, item_count = read_data(path)

    ftr = open(f'train_user_{fold}', 'wb')
    fte = open(f'test_user_{fold}', 'wb')
    train_users = []
    test_users = []
    for user in user_historicals:
        if random.random() < 0.9:
            train_users.append(user)
        else:
            test_users.append(user)

    pickle.dump(train_users, ftr)
    ftr.close()
    pickle.dump(test_users, fte)
    fte.close()


def compute_ndcg(labels, true_labels):
    """Compute Normalized Discounted Cumulative Gain"""
    dcg_labels = np.array(labels)
    dcg = np.sum(dcg_labels / np.log2(np.arange(2, dcg_labels.size + 2)))

    idcg_labels = np.array(true_labels)
    idcg = np.sum(idcg_labels / np.log2(np.arange(2, idcg_labels.size + 2)))
    if not idcg:
        return 0.

    res = dcg / idcg
    assert(res <= 1 and res >= 0)
    return res


def get_topk(action, k):
    """Get top-k items from action probabilities"""
    selection = np.argsort(action)[::-1][:k]
    return selection


def ppo_cold_train(file, test_num, warm_split, k, filter_nb=0):
    """Train model in cold-start setting using PPO"""
    user_history, user_size, item_size = read_data(file)

    if file == '1m':
        with open(f'1m_train_user_{test_num}', 'rb') as f:
            train_users = pickle.load(f)
        with open(f'1m_test_user_{test_num}', 'rb') as f:
            test_users = pickle.load(f)
    elif file == '100k':
        with open(f'train_user_{test_num}', 'rb') as f:
            train_users = pickle.load(f)
        with open(f'test_user_{test_num}', 'rb') as f:
            test_users = pickle.load(f)

    # Model hyperparameters
    lr = 0.001
    rnn_size = 100
    layer_size = 1
    embedding_dim = 100
    nb_epoch = 100
    slice_length = 20
    switch_epoch = 4
    
    # PPO specific hyperparameters
    ppo_clip_ratio = 0.2
    ppo_epochs = 4
    entropy_coef = 0.01
    value_coef = 0.5

    print(f"file:{file}, k:{k}, test_num:{test_num}, warm_split:{warm_split}")
    print(f"Using PPO with clip ratio:{ppo_clip_ratio}, epochs:{ppo_epochs}")

    # Initialize model with PPO
    im = PPOInteractiveModel(
        rnn_size, layer_size, item_size, embedding_dim, k, lr, device,
        ppo_clip_ratio=ppo_clip_ratio, ppo_epochs=ppo_epochs,
        entropy_coef=entropy_coef, value_coef=value_coef
    )
    im.to(device)
    print(time.asctime(time.localtime(time.time())))

    for epoch in range(nb_epoch):
        train_user_hit = []

        for j, user in enumerate(train_users):
            final_state = None
            user_selected_items = user_history[user]
            inference_length = len(user_selected_items)
            overall_length = inference_length
            start_slice = 0
            end_slice = start_slice + slice_length
            finish = False
            interest = get_cum_interesting(user_selected_items, item_size)
            masking = get_initial_masking(item_size)
            current_hit = []
            s_token = 0
            s_hit = 1.0

            while end_slice <= overall_length and start_slice < overall_length:
                slice_items = user_selected_items[start_slice: end_slice]
                inference_length = len(slice_items)
                if inference_length == 0:
                    break
                
                if epoch > switch_epoch:
                    # Use PPO reinforcement learning instead of standard policy gradient
                    rein, train_hit, final_state, masking, samples = im.ppo_reinforcement_learn(
                        interest, masking, inference_length, final_state, s_token, s_hit)
                else:
                    # Supervised learning remains the same
                    sup, train_hit, final_state, masking, samples = im.supervised_learn(
                        interest, masking, slice_items, inference_length, final_state, s_token, s_hit)

                start_slice = end_slice
                end_slice = start_slice + slice_length
                samples = samples.reshape(-1)
                train_hit = train_hit.reshape(-1)
                s_token = samples[-1]
                s_hit = -1.0
                if train_hit[-1] > 0:
                    s_hit = 1.0
                current_hit.extend(train_hit)

            train_user_hit.append(np.sum(current_hit) / overall_length)

        # Evaluation on test set
        test_mean_user_hit = []
        test_user_ndcg = []
        test_inter_diversity_list = []

        max_common_length = min([len(user_history[u]) - int(len(user_history[u]) * warm_split) for u in test_users])

        for user in test_users:
            final_state = None
            user_selected_items = user_history[user]
            inference_length = len(user_selected_items)

            warm_slice = int(inference_length * warm_split)
            warm_items = user_selected_items[:warm_slice]
            user_selected_items = user_selected_items[warm_slice:]

            inference_length = len(user_selected_items)

            interest = get_cum_interesting(user_selected_items, item_size)
            masking = get_masking(item_size, warm_items)
            
            # Run inference
            (user_item_probs, user_selected_items_pred, user_final_masking, 
             user_hit, user_immediate_reward, user_cumsum_reward) = im.inference(
                interest, masking, inference_length, final_state)

            user_item_probs = np.squeeze(user_item_probs, axis=0)
            current_ndcg = []
            current_masking = np.reshape(masking, (item_size,))
            user_hit = np.reshape(user_hit, (inference_length,))
            user_item_probs = np.reshape(user_item_probs, (inference_length, item_size))
            user_selected_items_pred = np.reshape(user_selected_items_pred, (inference_length,))

            interest_item_num = len(user_selected_items)

            for s in range(inference_length):
                if s == (max_common_length - 1):
                    prob = user_item_probs[s] * current_masking
                    topk_items = get_topk(prob, k).tolist()
                    test_inter_diversity_list.append(topk_items)

                true_labels = [0.0 for _ in range(k)]
                for ii in range(interest_item_num):
                    if ii >= k:
                        break
                    true_labels[ii] = 1.0

                ndcg_labels = []

                if user_hit[s] == 0.0:
                    current_ndcg.append(0.0)
                else:
                    prob = user_item_probs[s] * current_masking
                    topk_items = get_topk(prob, k)
                    for j in topk_items:
                        ndcg_label = 0.0
                        if j in user_selected_items:
                            ndcg_label = 1.0
                        ndcg_labels.append(ndcg_label)

                    current_ndcg.append(compute_ndcg(ndcg_labels, true_labels))

                    current_selected_item = user_selected_items_pred[s]

                    current_masking[current_selected_item] = 0.0
                    interest_item_num -= 1

            test_user_ndcg.append(np.mean(current_ndcg))
            current_hit_ratio = np.sum(user_hit) / inference_length

            test_mean_user_hit.append(current_hit_ratio)

        print(f"file:{file}, k:{k}, test_num:{test_num}, warm_split:{warm_split}")

        train_hit_mean = float(np.mean(train_user_hit))
        test_hit_mean = float(np.mean(test_mean_user_hit))
        test_ndcg_mean = float(np.mean(test_user_ndcg))
        diversity = inter_diversity(test_inter_diversity_list)

        print(f'epoch:{epoch}, train hr:{train_hit_mean:.4f}, test: HR = {test_hit_mean:.4f}, NDCG@10 = {test_ndcg_mean:.4f}, diversity={diversity:.4f}')

        # Save model checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': im.state_dict(),
                'optimizer_state_dict': im.optimizer.state_dict(),
            }, f'ppo_checkpoint_{file}_{test_num}_{epoch}.pt')


if __name__ == "__main__":
    file = '100k'

    for i in range(5):
        test_num = str(i)
        gen_100k_train_test(file, test_num)
        k = 10
        warm_split = 0.0
        ppo_cold_train(file, test_num, warm_split, k)