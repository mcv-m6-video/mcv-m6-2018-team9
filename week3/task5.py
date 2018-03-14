import numpy as np
import cv2

from data import cdnet
from video import bg_subtraction, morphology, shadow_detection
from evaluation import metrics, animations


def run(dataset):
    """Compute PR-curves and compare baseline with our best model in Week 3"""

    if dataset == 'highway':
        bsize = 50
        alpha_values = np.concatenate([np.linspace(0, 10, 30),
                                       np.linspace(11, 40, 10)])
        se_close = (15, 15)
        se_open = (4, 17)
        shadow_t1 = 0.082
        shadow_t2 = 0.017

    elif dataset == 'fall':
        bsize = 800
        alpha_values = np.concatenate([np.linspace(0, 10, 30),
                                       np.linspace(11, 40, 10)])
        se_close = (15, 15)
        se_open = (10, 10)

    elif dataset == 'traffic':
        bsize = 400
        alpha_values = np.concatenate([np.linspace(0, 10, 30),
                                      np.linspace(11, 40, 10)])
        se_open = (10, 10)
        shadow_t1 = 0.051
        shadow_t2 = 0.017

    # Model parameters
    rho = dict(highway=0.20, fall=0.10, traffic=0.15)

    # Read dataset
    train = cdnet.read_sequence('week3', dataset, 'train', colorspace='gray',
                                annotated=False)
    test, gt = cdnet.read_sequence('week3', dataset, 'test',
                                   colorspace='gray', annotated=True)
    test_c, __ = cdnet.read_sequence('week3', dataset, 'test', colorspace='rgb')

    # Adaptive model prediction
    model = bg_subtraction.create_model(train)

    # Compute AUC for Precision-Recall curves
    summaries_pred = []
    summaries_morph = []
    summaries_final = []

    for alpha in alpha_values:
        print(f"{alpha:0.4f}, ", end='', flush=True)
        pred = bg_subtraction.predict(test, model, alpha,
                                      rho=rho[dataset])
        filled8 = morphology.imfill(pred, neighb=8)
        clean = morphology.filter_small(filled8, bsize, neighb=4)

        if dataset == 'highway' or dataset == 'fall':
            #CLOSING
            st_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                se_close)
            morph_c = morphology.filter_morph(clean, cv2.MORPH_CLOSE,
                                              st_elem)

        #OPENING
        st_elem = cv2.getStructuringElement(cv2.MORPH_RECT, se_open)
        morph = morphology.filter_morph(morph_c, cv2.MORPH_OPEN,
                                        st_elem)

        if dataset == 'fall':
            final = morph  # shadow removal does not improve results
        else:
            shadowrem = shadow_detection.shadow_batch(test_c, shadow_t1, shadow_t2)
            final = np.logical_and(morph, shadowrem)

        summary_pred = metrics.eval_from_mask(pred, gt[:,0], gt[:,1])
        summary_morph = metrics.eval_from_mask(morph, gt[:,0], gt[:,1])
        summary_final = metrics.eval_from_mask(final, gt[:,0], gt[:,1])

        summaries_pred.append(summary_pred)
        summaries_morph.append(summary_morph)
        summaries_final.append(summary_final)

    # AUC results
    auc_pred = metrics.auc(dict(data=summaries_pred), 'prec-rec')
    auc_morph = metrics.auc(dict(data=summaries_morph), 'prec-rec')
    auc_best = metrics.auc(dict(data=summaries_final), 'prec-rec')
    print("\n")
    print("AUC for Morphology Closing "+str(se_close))
    print(f"AUC no imfill: {auc_pred:0.4f}")
    print(f"AUC Morphology: {auc_morph:0.4f}")
    print(f"AUC best: {auc_best:0.4f}")

    # More details for final model
    alpha1, prec1, alpha2, prec2 = metrics.find_extrema(
        summaries_final, alpha_values, metrics.prec)
    print("")
    print("Summaries for final model")
    print(f"Precision min: {prec1:0.3f} (alpha {alpha1:0.3f})")
    print(f"Precision max: {prec2:0.3f} (alpha {alpha2:0.3f})")

    alpha1, rec1, alpha2, rec2 = metrics.find_extrema(
        summaries_final, alpha_values, metrics.recall)
    print(f"Recall min: {rec1:0.3f} (alpha {alpha1:0.3f})")
    print(f"Recall max: {rec2:0.3f} (alpha {alpha2:0.3f})")

    alpha1, fsco1, alpha2, fsco2 = metrics.find_extrema(
        summaries_final, alpha_values, metrics.f1_score)
    print(f"F1-score min: {fsco1:0.3f} (alpha {alpha1:0.3f})")
    print(f"F1-score max: {fsco2:0.3f} (alpha {alpha2:0.3f})")

    # Plot PR-curves for pred / morph on the same plot
    plot_data = [dict(title=f"baseline (AUC={auc_pred:0.4f})",
                      data=summaries_pred),
                 # dict(title=f"morphology (AUC={auc_morph:0.4f})",
                 #      data=summaries_morph),
                 dict(title=f"best model (AUC={auc_best:0.4f})",
                      data=summaries_final)]
    metrics.plot_precision_recall_curves(plot_data, same_plot=True)
