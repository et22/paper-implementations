import numpy as np 

# code for implementing 'random walk' task described in figure 2
def sample_random_walk(n_states):
  # n_states is the number of non-terminal states
  state = int(np.floor(n_states/2))
  trajectory = [state]
  while state > -1 and state < n_states:
    if np.random.rand() < .5:
      state -= 1
    else:
      state += 1
    trajectory.append(state)

  if state == 5:
    outcome = 1
  else:
    outcome = 0

  return np.array(trajectory), outcome

def sample_training_set(n_samples, n_states):
  trajectories = []
  outcomes = []
  for i in range(n_samples):
    trajectory, outcome = sample_random_walk(n_states)
    trajectories.append(trajectory)
    outcomes.append(outcome)
  
  return trajectories, outcomes

def sample_training_sets(n_sets=100, n_samples=10, n_states=5):
  training_set = []
  for i in range(n_sets):
    training_set.append(sample_training_set(n_samples, n_states))
  return training_set

def random_walk_labels(n_states=5):
  # see equation 5 in sutton 1983
  # n_states is the number of non-terminal states
  Q = np.eye(n_states, n_states, k=1)*0.5 + np.eye(n_states, n_states, k=-1)*0.5
  h = np.zeros(n_states)
  h[-1] = 0.5

  return np.dot(np.linalg.inv(np.eye(n_states) - Q), h)

if __name__ == '__main__':
    n_states = 5
    training_sets = sample_training_sets(n_sets=100, n_samples=10, n_states=n_states)
    labels = random_walk_labels(n_states=n_states)