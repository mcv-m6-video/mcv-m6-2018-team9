import os
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from data import cdnet
from video import bg_subtraction
from evaluation import metrics
from evaluation import animations


OUT_VIDEO_PATH_BASE = os.path.join('week3', 'resources', 'prev')
OUT_VIDEO_EXTENSION = 'gif' #['gif','mp4']
OUT_VIDEO_CODEC = None #[None,'mp4v']


dataset_train_idx = {'highway': (1050, 1200),
                     'fall': (1460, 1510),
                     'traffic': (950, 1000)}

dataset_test_idx = {'highway': (1200, 1350),
                    'fall': (1510, 1560),
                    'traffic': (1000, 1050)}


def animate(test, model, f_results, beta, dataset, alpha_values, rho_values):

    best_rho, best_alpha = np.unravel_index(np.argmax(f_results),
                                            f_results.shape)

    pred = bg_subtraction.predict(test, model, alpha=alpha_values[best_alpha],
                                  rho=rho_values[best_rho])

    out_filename = 'adapt-' + dataset + '-alpha_' \
                   + str(alpha_values[best_alpha]) + '_rho_' \
                   + str(rho_values[best_rho])

    out_video_path_full = os.path.join(OUT_VIDEO_PATH_BASE, str(beta)) + '_'

    if out_filename is not None and OUT_VIDEO_EXTENSION is not None:
        print('>>>>Recording animation in ' + out_video_path_full +
              out_filename + '.' + OUT_VIDEO_EXTENSION + '...')

        animations.video_recorder(pred, out_video_path_full, out_filename,
                                  OUT_VIDEO_CODEC, OUT_VIDEO_EXTENSION)


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
    train_start, train_end = dataset_train_idx[dataset]
    test_start, test_end = dataset_test_idx[dataset]
    train = cdnet.read_dataset(dataset, train_start, train_end,
                               colorspace='gray', annotated=False)
    test, gt = cdnet.read_dataset(dataset, test_start, test_end,
                                  colorspace='gray', annotated=True)

    # Initial model
    model = bg_subtraction.create_model(train)
    model_backup = deepcopy(model)

    # Grid search
    f_results = np.zeros((len(rho_values), len(alpha_values)), dtype='float32')

    for i, rho in enumerate(rho_values):
        for j, alpha in enumerate(alpha_values):
            pred = bg_subtraction.predict(test, model, alpha, rho=rho)
            summary = metrics.eval_from_mask(pred, gt[:,0,:,:], gt[:,1,:,:])
            f_results[i, j] = metrics.f_score(summary, beta=beta)
            print('- alpha {:0.2f}, rho {:0.2f}: {:0.4f}'.format(
                alpha, rho, f_results[i, j]))

    animate(test, model_backup, f_results, beta, dataset, alpha_values, rho_values)
    return f_results


def run(dataset):
    """Task 2.1

    Grid search for hyperparameter selection in background estimation with
    adaptive model

    """
    betas = [10, 12, 14]

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
    for beta in betas:
        f_results = grid_search(dataset, alpha_values, rho_values, beta)

        # Find best score
        best_f = f_results.max()
        best_rho, best_alpha = np.unravel_index(np.argmax(f_results),
                                                f_results.shape)

        print('Best parameters for Beta= {:0.1f}: alpha {:0.2f}, rho {:0.2f}, '
              'with F-score {:0.4f}'.format(beta, alpha_values[best_alpha],
                                            rho_values[best_rho], best_f))

        # Plot results and print best parameters
        plot_grid_search(alpha_values, rho_values, f_results,
                         title=dataset)
