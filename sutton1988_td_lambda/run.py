import numpy as np
from utils import *
from td_lambda import *
from random_walk import *
from random_maze import *

def run_figure3_experiments(training_sets, labels, save_name):
    lambdas = list(np.linspace(0, 1, 21))
    alphas = np.linspace(0.05, 0.2, 3)
    errors = []
    opt_alphas = []
    for lamb in lambdas:
        out = [train_loop_conv(alpha, lamb, training_sets, labels) for alpha in alphas]
        opt_alphas.append(alphas[np.argmin(out)])
        errors.append(np.min(out))

    save_results({'lambdas': lambdas, 'errors': errors, 'opt_alphas': opt_alphas}, save_name)
    plot_xy_figure(lambdas, errors, "$\lambda$", "Error using best $\\alpha$", save_name=save_name)

def run_figure4_experiments(training_sets, labels, save_name, n_states=5):
    # train weights for figure 4
    lambdas = [0, 0.3, 0.8, 1]
    lambda_labels = [f"$\lambda={lambdas[i]}$" for i in range(len(lambdas))]

    alphas = np.linspace(0, 0.4, 13)
    error_alphas = []
    for lamb in lambdas:
        error_alphas.append([train_loop_once(alpha, lamb, training_sets, labels, n=n_states) for alpha in alphas])

    save_results({'lambdas': lambdas, 'error_alphas': error_alphas, 'alphas': alphas}, save_name)
    plot_xsys_figure([alphas for _ in range(len(lambdas))], error_alphas, "$\\alpha$", "Error", lambda_labels, save_name=save_name)

def run_figure5_experiments(training_sets, labels, save_name, n_states=5):
    # train weights for figure 5
    lambdas = list(np.linspace(0, 1, 21))

    alphas = np.linspace(0.01, 0.2, 5)
    errors = []
    opt_alphas = []

    for lamb in lambdas:
        out = [train_loop_once(alpha, lamb, training_sets, labels, n=n_states) for alpha in alphas]
        opt_alphas.append(alphas[np.argmin(out)])
        errors.append(np.min(out))
    
    save_results({'lambdas': lambdas, 'errors': errors, 'opt_alphas': opt_alphas}, save_name)
    plot_xy_figure(lambdas, errors, "$\lambda$", "Error using best $\\alpha$", save_name=save_name)

def run_main_experiments():
    n_states = 5

    training_sets = sample_training_sets(n_sets=100, n_samples=10, n_states=n_states)
    labels = random_walk_labels(n_states=n_states)

    run_figure3_experiments(training_sets, labels, save_name="figure3")
    run_figure4_experiments(training_sets, labels, save_name="figure4", n_states=n_states)
    run_figure5_experiments(training_sets, labels, save_name="figure5", n_states=n_states)

def run_num_state_experiments():
    for n_states in [3, 5, 7]:
        training_sets = sample_training_sets(n_sets=100, n_samples=10, n_states=n_states)
        labels = random_walk_labels(n_states=n_states)

        run_figure4_experiments(training_sets, labels, save_name=f"figure4_nstates={n_states}", n_states=n_states)
        run_figure5_experiments(training_sets, labels, save_name=f"figure5_nstates={n_states}", n_states=n_states)

def run_num_samples_experiments():
    for num_samples in [3, 3**2, 3**3, 3**4]:
        n_states = 5

        training_sets = sample_training_sets(n_sets=100, n_samples=num_samples, n_states=n_states)
        labels = random_walk_labels(n_states=n_states)

        run_figure4_experiments(training_sets, labels, save_name=f"figure4_nsamples={num_samples}", n_states=n_states)
        run_figure5_experiments(training_sets, labels, save_name=f"figure5_samples={num_samples}", n_states=n_states)

def summarize_num_samples_experiments():
    opt_alphas = []
    sample_list = [3, 3**2, 3**3, 3**4]
    for num_samples in sample_list:
        load_name=f"figure4_nsamples={num_samples}"
        results = load_results(load_name)

        lambdas = results['lambdas']
        errors = results['error_alphas'][lambdas == 0]
        alphas = results['alphas']

        opt_alpha = alphas[int(np.argmin(errors))]
        opt_alphas.append(opt_alpha)
    plot_xy_figure(sample_list, opt_alphas, "Number of sequences", "Optimal $\\alpha$ for $\lambda=0$", save_name="figure4_nsamples_summary")

    opt_lambdas = []
    sample_list = [3, 3**2, 3**3, 3**4]
    for num_samples in sample_list:
        load_name=f"figure5_samples={num_samples}"
        results = load_results(load_name)

        lambdas = results['lambdas']
        errors = results['errors']

        opt_lambda = lambdas[int(np.argmin(errors))]
        opt_lambdas.append(opt_lambda)
    plot_xy_figure(sample_list, opt_lambdas, "Number of sequences", "Optimal $\lambda$ with best $\\alpha$", save_name="figure5_nsamples_summary")

def run_maze_experiments():
    for max_depth in [1,2,3]:
        n_samples = 10
        n_states = np.sum([2 ** i for i in range(max_depth+1)]) - 1

        training_sets = sample_maze_training_sets(n_sets=100, n_samples=n_samples, max_depth=max_depth)
        labels = random_maze_labels(max_depth=max_depth)

        run_figure4_experiments(training_sets, labels, save_name=f"figure4_maze_depth={max_depth}_samples={n_samples}", n_states=n_states)
        run_figure5_experiments(training_sets, labels, save_name=f"figure5_maze_depth={max_depth}_samples={n_samples}", n_states=n_states)

    for n_samples in [3, 3 ** 2, 3 ** 3, 3 ** 4]:
        max_depth = 2
        n_states = np.sum([2 ** i for i in range(max_depth+1)]) - 1

        training_sets = sample_maze_training_sets(n_sets=100, n_samples=n_samples, max_depth=max_depth)
        labels = random_maze_labels(max_depth=max_depth)

        run_figure4_experiments(training_sets, labels, save_name=f"figure4_maze_depth={max_depth}_samples={n_samples}", n_states=n_states)
        run_figure5_experiments(training_sets, labels, save_name=f"figure5_maze_depth={max_depth}_samples={n_samples}", n_states=n_states)

if __name__ == '__main__':
    #run_main_experiments()
    #run_num_state_experiments()
    #run_num_samples_experiments()
    #summarize_num_samples_experiments()
    run_maze_experiments()
    pass

