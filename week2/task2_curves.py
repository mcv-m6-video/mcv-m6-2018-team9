import numpy as np

from data import cdnet
from video import bg_subtraction
from evaluation import metrics


dataset_train_idx = {'highway': (1050, 1200),
                     'fall': (1460, 1510),
                     'traffic': (950, 1000)}

dataset_test_idx = {'highway': (1200, 1350),
                    'fall': (1510, 1560),
                    'traffic': (1000, 1050)}

# Best rho parameters found by grid search
rho = {'highway': 0.10,
       'fall': 0.10,
       'traffic': 0.05}


def run(dataset):

    # Load datasets
    train_start, train_end = dataset_train_idx[dataset]
    test_start, test_end = dataset_test_idx[dataset]
    train = cdnet.read_dataset(dataset, train_start, train_end,
                               colorspace='gray', annotated=False)
    test, gt = cdnet.read_dataset(dataset, test_start, test_end,
                                  colorspace='gray', annotated=True)

    # Initial model
    model = bg_subtraction.create_model(train)

    # Predict with different alpha values (using best rho found previously with
    # grid search)
    results_list = []
    alpha_list = [1,2,3,4,5,6,7,8,9,10]

    for alpha in alpha_list:
        pred = bg_subtraction.predict(test, model, alpha, rho=rho[dataset])

        results = metrics.eval_from_mask(pred, gt[:,0], gt[:,1])
        results_list.append(results)

    # Best f1 score
    f1_scores = [metrics.f1_score(summary) for summary in results_list]
    best = np.argmax(f1_scores)
    print('Best alpha {:0.2f} with F1-score {:0.4f}'.format(
        alpha_list[best], f1_scores[best]))

    # Plot precision, recall and F1 curves w.r.t. alpha
    input_dict = dict(title='Adaptive modelling', data=results_list)
    metrics.plot_results_by_some_param([input_dict], alpha_list, 'alpha',
                                       'prec-rec-f1')

    # AUC for precision-recall curve
    auc = metrics.auc(input_dict, 'prec-rec')
    print('AUC for precision-recall curve:', auc)
