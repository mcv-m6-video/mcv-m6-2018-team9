import numpy as np

from data import cdnet
from video import bg_subtraction, morphology
from evaluation import metrics, animations


def run():

    experiments = []

    for dataset in ['highway', 'fall', 'traffic']:
        # Hyperparameters values to test
        if dataset == 'highway':
            alpha_values = [1, 2, 3, 4, 10, 20, 40]
            rho = 0.2
        elif dataset == 'fall':
            alpha_values = [1, 2, 3, 4, 10, 20, 40]
            rho = 0.1
        elif dataset == 'traffic':
            alpha_values = [1, 2, 3, 10, 40]
            rho = 0.15

        blob_sizes = range(0, 1001, 50)

        train = cdnet.read_sequence('week3', dataset, 'train', colorspace='gray',
                                    annotated=False)
        test, gt = cdnet.read_sequence('week3', dataset, 'test',
                                       colorspace='gray', annotated=True)

        # Adaptive model prediction
        model = bg_subtraction.create_model(train)
        # TODO: use parameters for each dataset

        data = []

        for bsize in blob_sizes:

            clean_datas = []

            for alpha in alpha_values:
                pred = bg_subtraction.predict(test, model, alpha, rho)

                filled4 = morphology.imfill(pred, neighb=4)

                clean_4 = morphology.filter_small(filled4, bsize, neighb=4)
                # animations.video_recorder(clean_4, '', 'w3t2_clean')
                clean_data = metrics.eval_from_mask(clean_4, gt[:,0], gt[:,1])
                clean_datas.append(clean_data)

            data.append(clean_datas)

        experiment = dict(title=dataset, data=data)

        experiments.append(experiment)

    metrics.plot_auc_by_removed_area(experiments, blob_sizes, alpha_values)

        # beta = 1
        #
        # fsco_pred = metrics.f_score(summary_pred, beta=beta)
        # fsco_fill4 = metrics.f_score(summary_fill4, beta=beta)
        # fsco_clean4 = metrics.f_score(summary_clean4, beta=beta)
        #
        # print('F-score (beta={}):'.format(beta))
        # print('\t no fill: {}'.format(fsco_pred))
        # print('\t fill4: {}'.format(fsco_fill4))
        # print('\t clean4: {}'.format(fsco_clean4))

