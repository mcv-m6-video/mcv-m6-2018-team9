from data import cdnet
from video import bg_subtraction
from evaluation import metrics



# Ground truth labels

LABEL_STATIC = 0
LABEL_SHADOW = 50
LABEL_OUTSIDE_ROI = 85
LABEL_UNKWN_MOTION = 170
LABEL_MOTION = 255

def run():

    # Fit the model to the first half of the images
    ims_train = cdnet.read_dataset('highway', 1, 1200, colorspace='gray',
                                   annotated=False)
    model = bg_subtraction.create_model(ims_train)

    # Test the model with the second half of the images
    ims, gts = cdnet.read_dataset('highway', 1200, 1350,
                                  colorspace='gray', annotated=True, bg_th=LABEL_SHADOW, fg_th=LABEL_MOTION)
    results_list = []
    alpha_list = [1,3]
    print(len(alpha_list))
    for alpha in alpha_list:
        pred = bg_subtraction.predict(ims, model, alpha)

        #import pdb; pdb.set_trace()
        # Extract metrics (TP, FP, ...) and plot results
        results = metrics.eval_from_mask(pred, gts[:,0], gts[:,1])
        results_list.append(dict(description='Test A (Gaussian Modelling)', data=results))

    print(len(results_list))
    metrics.plot_results_by_some_param(results_list, alpha_list, 'alpha',
                                       'prec-rec')






