import os.path
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from video import optical_flow
from evaluation import metrics
from matplotlib.ticker import FuncFormatter
import cv2
import imageio
import numpy.ma as ma

from skimage.transform import warp
from skimage.transform import SimilarityTransform
from skimage.transform import warp
from skimage.io import imsave, imread, imshow
from skimage.color import rgb2gray
from skimage.viewer import ImageViewer
from data import cdnet
from evaluation import multigif
from video import bg_subtraction, morphology, shadow_detection

def run():
    
    dataset = 'traffic'
    bsize = 400
    alpha_values = np.concatenate([np.linspace(0, 10, 50),
                                  np.linspace(11, 40, 15)])
    se_close = (15, 15)
    #diagonal width
    k = 9
    #diagonal height
    l = 30
    se_open = np.eye(l, dtype=np.uint8)
    for r in range(0, k):
        se_open = np.logical_or(se_open,
                                np.eye(l, dtype=np.uint8, k=r + 1))
        se_open = np.logical_or(se_open,
                                np.eye(l, dtype=np.uint8, k=r - 1))
    se_open = np.transpose(se_open.astype(np.uint8))
    shadow_t1 = 0.051
    shadow_t2 = 0.017

    # Model parameters
    rho = dict(highway=0.20, fall=0.10, traffic=0.15)

    # Read dataset
    train = cdnet.read_dataset('traffic', 950, 1000, colorspace='rgb', annotated=False)
    test, gt = cdnet.read_dataset('traffic', 1000, 1050, colorspace='rgb', annotated=True)

    test_c, gts = cdnet.read_dataset('traffic', 1000, 1050, colorspace='rgb', annotated=True)


    #Stabilize train and test images
    stab_train, stab_masks = optical_flow.stabilize(train, mode= 'f')
    stab_test, stab_masks_test, new_gt = optical_flow.stabilize(test_c, gts, mode= 'f')

    valid_mask = new_gt[:,1].copy()
    masks_for_process = np.logical_and(valid_mask, stab_masks_test)


    # Adaptive model prediction
    model = bg_subtraction.create_model(train)
    model_mask = bg_subtraction.create_model_mask(stab_train, stab_masks)

    # Compute AUC for Precision-Recall curves
    summaries_pred = []
    summaries_morph = []
    summaries_final = []
    summaries_stab = []

    for alpha in alpha_values:
        #print(f"{alpha:0.4f}, ", end='', flush=True)
        print("alpha: ", alpha)
        pred = bg_subtraction.predict(test_c, model, alpha,
                                      rho=rho[dataset])

        stab_pred = bg_subtraction.predict_masked(stab_test, stab_masks_test, model_mask, alpha,
                                      rho=rho[dataset])


        filled8 = morphology.imfill(pred, neighb=8)
        filled8_stab = morphology.imfill(stab_pred, neighb=8)

        clean = morphology.filter_small(filled8, bsize, neighb=4)
        clean_stab = morphology.filter_small(filled8_stab, bsize, neighb=4)

        #CLOSING
        st_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, se_close)
        morph_c = morphology.filter_morph(clean, cv2.MORPH_CLOSE, st_elem)
        morph_c_stab = morphology.filter_morph(clean_stab, cv2.MORPH_CLOSE, st_elem)

        #OPENING
        if(dataset == 'traffic'):
            st_elem = se_open
        else:
            st_elem = cv2.getStructuringElement(cv2.MORPH_RECT, se_open)

        morph = morphology.filter_morph(morph_c, cv2.MORPH_OPEN, st_elem)
        morph_stab = morphology.filter_morph(morph_c_stab, cv2.MORPH_OPEN, st_elem)
        if dataset == 'fall':
            final = morph  # shadow removal does not improve results
        else:
            shadowrem = shadow_detection.shadow_batch(test_c, shadow_t1, shadow_t2)
            #shadowrem_stab = shadow_detection.shadow_batch(stab_test, shadow_t1, shadow_t2)
            final = np.logical_and(morph, shadowrem)
            final_stab = morph_stab


        summary_pred = metrics.eval_from_mask(final, gt[:,0], gt[:,1])
        summary_morph = metrics.eval_from_mask(morph, gt[:,0], gt[:,1])
        summary_final = metrics.eval_from_mask(final, gt[:,0], gt[:,1])
        summary_stab = metrics.eval_from_mask(final_stab, new_gt[:,0], new_gt[:,1])

        summaries_pred.append(summary_pred)
        summaries_morph.append(summary_morph)
        summaries_final.append(summary_final)
        summaries_stab.append(summary_stab)

    # AUC results
    auc_pred = metrics.auc(dict(data=summaries_pred), 'prec-rec')
    auc_morph = metrics.auc(dict(data=summaries_morph), 'prec-rec')
    auc_best = metrics.auc(dict(data=summaries_final), 'prec-rec')
    auc_stab = metrics.auc(dict(data=summaries_stab), 'prec-rec')
    print("\n")
    print("AUC for Morphology Closing "+str(se_close))
    print("AUC no imfill: ", auc_pred)
    print("AUC Morphology: ", auc_morph)
    print("AUC old best: ", auc_best)
    print("AUC stabilized: ", auc_stab)

    # More details for final model
    alpha1, prec1, alpha2, prec2 = metrics.find_extrema(summaries_stab, alpha_values, metrics.prec)
    print("")
    print("Summaries for final model")
    print("Precision min: ", prec1, " alpha ", alpha1)
    print("Precision max: ", prec2, " alpha ", alpha2)

    alpha1, rec1, alpha2, rec2 = metrics.find_extrema(summaries_stab, alpha_values, metrics.recall)
    print("Recall min: ", rec1,  "alpha ", alpha1)
    print("Recall max: ", rec2,  "alpha ", alpha2)

    alpha1, fsco1, alpha2, fsco2 = metrics.find_extrema(summaries_stab, alpha_values, metrics.f1_score)
    print("F1-score min: ", fsco1, "alpha ", alpha1)
    print("F1-score max: ", fsco2, "alpha ", alpha2)

    # Plot PR-curves for pred / morph on the same plot
    plot_data = [dict(title="stabilized video",data=summaries_stab), dict(title="Raw video ", data=summaries_pred)]
    metrics.plot_precision_recall_curves(plot_data, same_plot=True)