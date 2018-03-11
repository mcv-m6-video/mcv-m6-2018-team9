import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from data import cdnet
from video import bg_subtraction
from evaluation import metrics


def plot_grid_search(alpha_values, rho_values, f1_results, title=''):
    """Plot F1-score for a grid search"""
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # https://matplotlib.org/examples/mplot3d/surface3d_demo.html
    X, Y = np.meshgrid(alpha_values, rho_values)
    surf = ax.plot_surface(X, Y, f1_results, cmap='plasma',
                           linewidth=0, antialiased=True)
    fig.colorbar(surf)

    ax.set_xlabel('alpha', size='large', weight='bold')
    ax.set_ylabel('rho', size='large', weight='bold')
    ax.set_zlabel('f1', size='large', weight='bold')

    plt.title(title)
    plt.show()

    return fig


def grid_search(dataset, alpha_values, rho_values, beta):
    """Perform a grid search over rho and alpha hyperparameters"""
    # Load datasets
    train = cdnet.read_sequence('week3', dataset, 'train',
                                colorspace='gray', annotated=False)
    test, gt = cdnet.read_sequence('week3', dataset, 'test',
                                   colorspace='gray', annotated=True)

    # Initial model
    model = bg_subtraction.create_model(train)

    # Grid search
    f_results = np.zeros((len(rho_values), len(alpha_values)), dtype='float32')

    for i, rho in enumerate(rho_values):
        for j, alpha in enumerate(alpha_values):
            pred = bg_subtraction.predict(test, model, alpha, rho=rho)
            summary = metrics.eval_from_mask(pred, gt[:,0], gt[:,1])
            f_results[i, j] = metrics.f_score(summary, beta=beta)
            print('- alpha {:0.2f}, rho {:0.2f}: {:0.4f}'.format(
                alpha, rho, f_results[i, j]))

    return f_results


def run(dataset):
    """Task 1 - Choose the best configuration from week 2

    Performs a grid search for hyperparameter selection in background
    estimation with adaptive model. The criterium to choose the best model is
    the highest F-score.

    """
    # Hyperparameters values to test
    if dataset == 'highway':
        alpha_values = np.arange(0, 8, 0.25)
        rho_values = np.arange(0, 1, 0.10)
    elif dataset == 'fall':
        alpha_values = np.arange(0, 8, 0.25)
        rho_values = np.arange(0, 1, 0.10)
    elif dataset == 'traffic':
        alpha_values = np.arange(0, 6, 0.20)
        rho_values = np.arange(0, 0.4, 0.05)

    # Grid search
    beta = 2  # beta for F-score formula
    f_results = grid_search(dataset, alpha_values, rho_values, beta)

    # Find best score
    best_f = f_results.max()
    best_rho, best_alpha = np.unravel_index(np.argmax(f_results),
                                            f_results.shape)

    print('Best parameters for Beta={:0.1f}: alpha {:0.2f}, rho {:0.2f}, '
          'with F-score {:0.4f}'.format(beta, alpha_values[best_alpha],
                                        rho_values[best_rho], best_f))

    # Plot results and print best parameters
    plot_grid_search(alpha_values, rho_values, f_results,
                     title=dataset)
