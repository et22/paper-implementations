from copy import deepcopy
import numpy as np

# td update - sum term
def compute_td_sum(trajectory, lamb, n):
  init_sum = np.zeros(n)
  init_sum[trajectory[0]] = 1
  td_sums = [init_sum]

  for i in range(1, len(trajectory)-1):
    del_wP = np.zeros(n)
    del_wP[trajectory[i]] = 1

    td_sums.append(lamb * td_sums[-1] + del_wP)

  return np.array(td_sums)

def td_update(trajectory, outcome, alpha, lamb, weights, n):
  delta_weights = np.zeros_like(weights)

  td_sums = compute_td_sum(trajectory, lamb, n)
  for t in range(len(trajectory)-1):
    pred_t = weights[trajectory[t]]

    if t == len(trajectory) - 2:
      pred_t1 = outcome
    else:
      pred_t1 = weights[trajectory[t+1]] 
    delta_weights += alpha * (pred_t1 - pred_t) * td_sums[t]
  
  return delta_weights

# rmse error
def rmse(labels, preds):
  return np.sqrt(np.sum((labels-preds) ** 2)/len(preds))

# training loop - only do one pass through 10 sequences to train weights
def train_loop_once(alpha, lamb, training_sets, labels, n=5):
  rmses = []
  for training_set in training_sets:
    weights = np.ones(n) * 0.5
    for (trajectory, outcome) in zip(training_set[0], training_set[1]):
      weights += td_update(trajectory, outcome, alpha, lamb, weights, n=n)

    rmses.append(rmse(labels, weights))

  return np.mean(rmses)

# training loop - train until convergence on 10 sequences to train weights
def train_loop_conv(alpha, lamb, training_sets, labels, n=5, max_iter = 100, eps=1e-2):
  rmses = []
  for training_set in training_sets:
    weights = np.ones(n) * 0.5
    prev_weights = np.ones(n)
    iter_cnt = 0
    while rmse(prev_weights, weights) > eps and iter_cnt < max_iter:
      iter_cnt += 1
      prev_weights = deepcopy(weights)

      delta_weights = np.zeros_like(weights)
      for (trajectory, outcome) in zip(training_set[0], training_set[1]):
        delta_weights += td_update(trajectory, outcome, alpha, lamb, weights, n=n)

      weights = weights + delta_weights/len(training_set[1])

    rmses.append(rmse(labels, weights))

  return np.mean(rmses)