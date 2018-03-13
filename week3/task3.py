import numpy as np
import matplotlib.pyplot as plt
import cv2

from data import cdnet
from video import bg_subtraction, morphology
from evaluation import metrics, animations


def run(dataset):
    """Compute PR-curves and compare the 4 methods (pred, fill8,
    area filtering, morphology)"""

    if dataset == 'highway':
        bsize = 50
        alpha_values = np.concatenate([np.linspace(0, 10, 30),
                                       np.linspace(11, 40, 10)])
        se_close = (15, 15)
        se_open = (4, 17)

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

    # Model parameters
    rho = dict(highway=0.20, fall=0.10, traffic=0.15)

    # Read dataset
    train = cdnet.read_sequence('week3', dataset, 'train', colorspace='gray',
                                annotated=False)
    test, gt = cdnet.read_sequence('week3', dataset, 'test',
                                   colorspace='gray', annotated=True)

    # Adaptive model prediction
    model = bg_subtraction.create_model(train)

    # Compute AUC for Precision-Recall curves
    summaries_pred = []
    summaries_fill8 = []
    summaries_clean = []
    summaries_morph = []

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

        summary_pred = metrics.eval_from_mask(pred, gt[:,0], gt[:,1])
        summary_fill8 = metrics.eval_from_mask(filled8, gt[:,0], gt[:,1])
        summary_clean = metrics.eval_from_mask(clean, gt[:,0], gt[:,1])
        summary_morph = metrics.eval_from_mask(morph, gt[:,0], gt[:,1])

        summaries_pred.append(summary_pred)
        summaries_fill8.append(summary_fill8)
        summaries_clean.append(summary_clean)
        summaries_morph.append(summary_morph)

    # AUC results
    auc_pred = metrics.auc(dict(data=summaries_pred), 'prec-rec')
    auc_fill8 = metrics.auc(dict(data=summaries_fill8), 'prec-rec')
    auc_clean = metrics.auc(dict(data=summaries_clean), 'prec-rec')
    auc_morph = metrics.auc(dict(data=summaries_morph), 'prec-rec')
    print("\n")
    print("AUC for Morphology Closing "+str(se_close))
    print(f"AUC no imfill: {auc_pred:0.4f}")
    print(f"AUC imfill 8-neighb: {auc_fill8:0.4f}")
    print(f"AUC area filter blob {str(bsize):4}: {auc_clean:0.4f}")
    print(f"AUC Morphology: {auc_morph:0.4f}")

    # More details for morphology
    alpha1, prec1, alpha2, prec2 = metrics.find_extrema(
        summaries_morph, alpha_values, metrics.prec)
    print("")
    print("Summaries for Morphology")
    print(f"Precision min: {prec1:0.3f} (alpha {alpha1:0.3f})")
    print(f"Precision max: {prec2:0.3f} (alpha {alpha2:0.3f})")

    alpha1, rec1, alpha2, rec2 = metrics.find_extrema(
        summaries_morph, alpha_values, metrics.recall)
    print(f"Recall min: {rec1:0.3f} (alpha {alpha1:0.3f})")
    print(f"Recall max: {rec2:0.3f} (alpha {alpha2:0.3f})")

    alpha1, fsco1, alpha2, fsco2 = metrics.find_extrema(
        summaries_morph, alpha_values, lambda x: metrics.f_score(x,
                                                                beta=1))
    print(f"F2-score min: {fsco1:0.3f} (alpha {alpha1:0.3f})")
    print(f"F2-score max: {fsco2:0.3f} (alpha {alpha2:0.3f})")

    # Plot PR-curves for pred / fill4 / fill8 on the same plot
    plot_data = [dict(title=f"no imfill (AUC={auc_pred:0.4f})",
                      data=summaries_pred),
                 dict(title=f"imfill8 (AUC={auc_fill8:0.4f})",
                      data=summaries_fill8),
                 dict(title=f"area filter blob {str(bsize):4}\
                                (AUC={auc_clean:0.4f})",
                      data=summaries_clean),
                 dict(title=f"morphology: (AUC={auc_morph:0.4f})",
                      data=summaries_morph),
                 ]












