import random
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import the PPO model instead of original InteractiveModel
from PPOInteractive_Model import get_cum_interesting, get_initial_masking, get_masking, PPOExternalMemInteractiveModel

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")


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


def gen_1m_train_test(path, fold):
    """Generate train-test split for MovieLens 1M dataset"""
    user_historicals, user_count, item_count = read_data(path)

    ftr = open(f'1m_train_user_{fold}', 'wb')
    fte = open(f'1m_test_user_{fold}', 'wb')
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

    return dcg / idcg


def get_topk(action, k):
    """Get top-k items from action probabilities"""
    selection = np.argsort(action)[::-1][:k]
    return selection

def visualize_training(train_hits, test_hits, checkpoint_epochs=None):
    """
    Visualize training metrics with customizable plots
    
    Args:
        train_hits: List of train hit rates per epoch
        test_hits: List of test hit rates per epoch
        checkpoint_epochs: Optional list of epochs where checkpoints were saved
    """
    plt.figure(figsize=(12, 8))
    
    # Plot train hit rates
    plt.subplot(2, 1, 1)
    
    epochs = range(1, len(train_hits) + 1)
    plt.plot(epochs, train_hits, 'b-', marker='o', markersize=5, label='Training Hit Rate')
    
    # Add markers for checkpoint epochs if provided
    if checkpoint_epochs:
        checkpoint_train_hits = [train_hits[e-1] for e in checkpoint_epochs if e <= len(train_hits)]
        checkpoint_x = [e for e in checkpoint_epochs if e <= len(train_hits)]
        plt.plot(checkpoint_x, checkpoint_train_hits, 'ro', markersize=8, label='Checkpoints')
    
    plt.title('Training Hit Rate Over Time', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Hit Rates', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Add moving average line for trend
    window_size = min(5, len(train_hits))
    if window_size > 1:
        moving_avg = np.convolve(train_hits, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size, len(train_hits) + 1), moving_avg, 'g-', linewidth=2, 
                 label=f'{window_size}-Epoch Moving Avg')
        plt.legend()
    
    # Plot test hit rates
    plt.subplot(2, 1, 2)
    plt.plot(epochs, test_hits, 'g-', marker='o', markersize=5, label='Test Hit Rates')
        
    # Add markers for checkpoint epochs if provided
    if checkpoint_epochs:
        checkpoint_accs = [test_hits[e-1] for e in checkpoint_epochs if e <= len(test_hits)]
        checkpoint_x = [e for e in checkpoint_epochs if e <= len(test_hits)]
        plt.plot(checkpoint_x, checkpoint_accs, 'ro', markersize=8, label='Checkpoints')
        
    plt.title('Test Hit Rates Over Time', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Hit Rates', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"warm_training_visualization{test_num}.png")
    plt.show()
    
    print(f"Training visualization saved to 'warm_training_visualization{test_num}.png'")

def ppo_warm_train(file, test_num, warm_split, k):
    """Train model in warm-start setting with memory using PPO"""
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
    lr = 0.0001
    rnn_size = 100
    layer_size = 1
    embedding_dim = 100
    nb_epoch = 50
    slice_length = 20
    switch_epoch = 10
    kl_coef = 0.005
    
    # PPO specific hyperparameters
    ppo_clip_ratio = 0.1
    ppo_epochs = 4
    entropy_coef = 0.001
    value_coef = 0.5

    print(f"file:{file}, test_num:{test_num}, k:{k}, warm_split:{warm_split}")
    print(f"Using PPO with clip ratio:{ppo_clip_ratio}, epochs:{ppo_epochs}")
    print(f'train_users: {len(train_users)}')
    print(f'Total users: {len(test_users) + len(train_users)}')
    print(f'avg length: {np.mean([len(user_history[u]) for u in train_users])}')
    print(f'item_num: {item_size}\n')

    # Initialize model with PPO
    im = PPOExternalMemInteractiveModel(
        rnn_size, layer_size, item_size, embedding_dim, k, lr, device,
        kl_coef, ppo_clip_ratio=ppo_clip_ratio, ppo_epochs=ppo_epochs,
        entropy_coef=entropy_coef, value_coef=value_coef
    )
    im.to(device)

    for epoch in range(nb_epoch):
        train_mean_user_hit = []

        for j, user in enumerate(train_users):
            final_state = None
            user_selected_items = user_history[user]
            inference_length = len(user_selected_items)
            overall_length = inference_length

            warm_slice = int(overall_length * warm_split)
            warm_items = user_selected_items[:warm_slice]
            user_selected_items = user_selected_items
            overall_length = len(user_selected_items)

            start_slice = 0
            end_slice = start_slice + slice_length

            mem = get_cum_interesting(warm_items, item_size)
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
                    loss, rein, train_hit, final_state, masking, samples = im.ppo_reinforcement_learn(
                        interest, masking, mem, inference_length, final_state, s_token, s_hit)
                else:
                    # Supervised learning remains the same
                    loss, sup, train_hit, final_state, masking, samples = im.supervised_learn(
                        interest, masking, mem, slice_items, inference_length, final_state, s_token, s_hit)

                start_slice = end_slice
                end_slice = start_slice + slice_length
                samples = samples.reshape(-1)
                train_hit = train_hit.reshape(-1)
                s_token = samples[-1]
                s_hit = -1.0

                if train_hit[-1] > 0:
                    s_hit = 1.0
                current_hit.extend(train_hit)
            
            train_mean_user_hit.append(np.sum(current_hit) / overall_length)

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

            mem = get_cum_interesting(warm_items, item_size)
            interest = get_cum_interesting(user_selected_items, item_size)
            masking = get_masking(item_size, warm_items)

            s_token = 0
            
            # Run inference
            (user_item_probs, user_selected_items_pred, user_final_masking, 
             user_hit, user_immediate_reward, user_cumsum_reward) = im.inference(
                interest, masking, mem, inference_length, final_state, s_t=s_token)

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

            user_hit = user_hit.reshape(-1)
            samples = user_selected_items_pred.reshape(-1)

            current_hit_ratio = np.sum(user_hit) / inference_length
            test_mean_user_hit.append(current_hit_ratio)

        train_hit_mean = float(np.mean(train_mean_user_hit))
        print(f"file:{file}, test_num:{test_num}, k:{k}, warm_split:{warm_split}")
        test_hit_mean = float(np.mean(test_mean_user_hit))
        test_ndcg_mean = float(np.mean(test_user_ndcg))
        diversity = inter_diversity(test_inter_diversity_list)

        print(f'epoch:{epoch}, train hr:{train_hit_mean:.4f}, test: HR = {test_hit_mean:.4f}, NDCG@10 = {test_ndcg_mean:.4f}, diversity = {diversity:.4f}\n')

        # Save model checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': im.state_dict(),
                'optimizer_state_dict': im.optimizer.state_dict(),
            }, f'ppo_warm_checkpoint_{file}_{test_num}_{epoch}.pt')


if __name__ == "__main__":
    file = '100k'
    k = 10
    warm_splits = 0.5
    train_hits = []
    test_hits = []

    for i in range(5):
        test_num = str(i)
        gen_100k_train_test(file, test_num)
        warm_split = 0.5  # Using the warm_splits value
        ppo_warm_train(file, test_num, warm_split, k)
        visualize_training(train_hits,test_hits)