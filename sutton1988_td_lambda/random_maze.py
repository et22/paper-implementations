import numpy as np 

# code for implementing 'binary maze' task
def sample_random_maze(max_depth, p_terminate=0.05):
  # n_states is the number of non-terminal states
  target_leaf = (2 ** max_depth) - 1
  state = (0, 0)
  trajectory = [state]
  num_steps = 0

  while state != (max_depth, target_leaf) and state != (-1, -1):
    num_steps += 1
    if np.random.rand() < p_terminate:
        state = (-1, -1)
    else:
        if state[0] > 0 and state[0] < max_depth:
            if np.random.rand() < 2/3:
                state = (state[0] + 1, int(2*state[1] + np.random.rand() < 0.5))          
            else:
                state = (state[0] - 1, int(np.floor(state[1]/2)))
        elif state[0] == max_depth:
            state = (state[0] - 1, int(np.floor(state[1]/2)))
        else:
            state = (state[0] + 1, int(2*state[1] + np.random.rand() < 0.5))          

    trajectory.append(state)

  if state == (max_depth, target_leaf):
    outcome = 1
  else:
    outcome = 0

  state_encoding = []
  for state in trajectory:
    state_encoding.append(int(2 ** state[0] + state[1] - 1))

  return np.array(state_encoding), outcome

def sample_maze_training_set(n_samples, max_depth):
  trajectories = []
  outcomes = []
  for i in range(n_samples):
    trajectory, outcome = sample_random_maze(max_depth)
    trajectories.append(trajectory)
    outcomes.append(outcome)
  
  return trajectories, outcomes

def sample_maze_training_sets(n_sets=100, n_samples=10, max_depth=2):
  training_set = []
  for i in range(n_sets):
    training_set.append(sample_maze_training_set(n_samples, max_depth))
  return training_set

def random_maze_labels(max_depth=2, p_terminate=0.05):
  # see equation 5 in sutton 1983
  # n_states is the number of non-terminal states
  n_states = np.sum([2 ** i for i in range(max_depth+1)])
  Q = np.zeros((n_states, n_states))

  h = np.zeros(n_states-1)
  for i in range(max_depth+1):
    for j in range(2 ** i):
        if i == 0 and j == 0:
            Q[0, 1] = 1/2
            Q[0, 2] = 1/2
        elif i < max_depth:
            Q[2 ** i + j - 1, 2 ** (i-1) + int(np.floor(j/2)) - 1] = 1/3
            Q[2 ** i + j - 1, 2 ** (i+1) + 2*j - 1] = 1/3
            Q[2 ** i + j - 1, 2 ** (i+1) + 2*j] = 1/3
        else:
            Q[2 ** i + j - 1, 2 ** (i-1) + int(np.floor(j/2)) - 1] = 1

            if i == max_depth and j == (2 ** max_depth) - 1:
                h[2 ** (i-1) + int(np.floor(j/2)) - 1] = 1/3

  Q = (1-p_terminate) * Q # 5% chance of terminating in each state
  Q = Q[:-1, :-1] # exclude the terminal 'reward' state  

  return np.dot(np.linalg.inv(np.eye(n_states-1) - Q), h)

if __name__ == '__main__':
    max_depth = 2
    training_sets = sample_maze_training_sets(n_sets=100, n_samples=10, max_depth=2)
    labels = random_maze_labels(max_depth=max_depth)