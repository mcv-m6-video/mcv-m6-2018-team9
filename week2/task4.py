from data import cdnet
from video import bg_subtraction
from evaluation import metrics



# Ground truth labels

LABEL_STATIC = 0
LABEL_SHADOW = 50
LABEL_OUTSIDE_ROI = 85
LABEL_UNKWN_MOTION = 170
LABEL_MOTION = 255

dataset_train_idx = {'highway': (1050, 1200),
                     'fall': (1460, 1510),
                     'traffic': (950, 1000)}

dataset_test_idx = {'highway': (1200, 1350),
                    'fall': (1510, 1560),
                    'traffic': (1000, 1050)}

def run():

    for ds in ['highway', 'fall', 'traffic']:
        input('Go to ' + ds)
        # Fit the model to the first half of the images
        ims_train = cdnet.read_dataset(ds, dataset_train_idx[ds][0],
                                       dataset_train_idx[ds][1],
                                       colorspace='ycbcr-only-color',
                                       annotated=False)
        model = bg_subtraction.create_model(ims_train)

        # Test the model with the second half of the images
        ims, gts = cdnet.read_dataset(ds, dataset_test_idx[ds][0],
                                      dataset_test_idx[ds][1],
                                      colorspace='ycbcr-only-color',
                                      annotated=True,
                                      bg_th=LABEL_SHADOW, fg_th=LABEL_MOTION)
        results_list = []
        results_dicts = []
        alpha_list = [1,2,3,4,5,6,7,8,9,10]
        for alpha in alpha_list:
            pred = bg_subtraction.predict(ims, model, alpha)

            #import pdb; pdb.set_trace()
            # Extract metrics (TP, FP, ...) and plot results
            results = metrics.eval_from_mask(pred, gts[:,0], gts[:,1])
            results_list.append(results)

        results_dicts.append(dict(title='CbCr test for ' + ds,
                                  data=results_list))

        metrics.plot_results_by_some_param(results_dicts, alpha_list, 'alpha',
                                           'F1')

        metrics.plot_precision_recall_curves(results_dicts)

        area_under_curve = metrics.auc(results_dicts[0], 'prec-rec')

        print('Area Under the Curve (precision-recall): ' + str(area_under_curve))
