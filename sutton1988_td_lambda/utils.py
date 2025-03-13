import matplotlib.pyplot as plt
import pickle 
import numpy as np

# code for plotting figures 3-5
def plot_xsys_figure(xs, ys, xlabel, ylabel, labels, save_name):
  plt.figure(figsize=(4, 3.5))
  for x, y, label in zip(xs, ys, labels):
    inc_idx = np.array(y) < .7
    x = np.array(x)
    y = np.array(y)
    x = x[inc_idx]
    y = y[inc_idx]
    plot_xy(x, y, xlabel, ylabel)
    plt.text(x[-1] + .05*x[-1], y[-1], label, fontsize=10)

  xl = plt.gca().get_xlim()
  plt.xlim(xl[0], xl[1]*1.25)
  yl = plt.gca().get_ylim()
  plt.ylim(yl[0], yl[1]*1.1)
  plt.tight_layout()
  plt.savefig("./figures/" + save_name + ".png")

def plot_xy_figure(x, y, xlabel, ylabel, save_name):
  plt.figure(figsize=(4, 3.5))
  plot_xy(x, y, xlabel, ylabel)
  plt.tight_layout()
  plt.savefig("./figures/" + save_name + ".png")

def plot_xy(x, y, xlabel, ylabel):
  plt.plot(x, y, 'k-', linewidth=1, marker='.')
  plt.xlabel(xlabel, fontsize=12)
  plt.ylabel(ylabel, fontsize=12)

def save_results(results, save_name):
  with open(f"./results/{save_name}.pkl", "wb") as f:
    pickle.dump(results, f)

def load_results(load_name):
  with open(f"./results/{load_name}.pkl", "rb") as f:
    return pickle.load(f)