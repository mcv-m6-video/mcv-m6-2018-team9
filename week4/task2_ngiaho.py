#script to perform video stabilization based on AdamSpannbauer implementation.
#	INPUTS: IMAGE SEQUENCE
#	OUTPUTS: STABELIZED VIDEO AVI, TRANSFORMATION DF CSV, SMOOTHED TRAJECTORY CSV

#RESOURCE & ORIGINAL CPP AUTHOR OF THIS VIDEO STAB LOGIC: http://nghiaho.com/?p=2093
#ORIGINAL CPP: http://nghiaho.com/uploads/code/videostab.cpp

import numpy as np
import cv2
from data import cdnet
from evaluation import metrics, animations
from video import bg_subtraction, morphology, video_stabilization
import matplotlib.pyplot as plt
import time


compareOutput = 0
maxWidth = 320
out_path = '.'


def run(dataset):

    # Model & Morphologyparameters
    if dataset == 'traffic':
        bsize = 400
        alpha_values = np.concatenate([np.linspace(0, 10, 30),
                                       np.linspace(11, 40, 10)])
        se_close = (15, 15)
        k = 9
        l = 30
        se_open = np.eye(l, dtype=np.uint8)
        for r in range(0, k):
            se_open = np.logical_or(se_open,
                                    np.eye(l, dtype=np.uint8, k=r + 1))
            se_open = np.logical_or(se_open,
                                    np.eye(l, dtype=np.uint8, k=r - 1))
        se_open = np.transpose(se_open.astype(np.uint8))

    rho = dict(highway=0.20, fall=0.10, traffic=0.15)

    # Read dataset
    train, gt_t = cdnet.read_sequence('week4', dataset, 'train',
                                colorspace='gray',annotated=True)
    test, gt = cdnet.read_sequence('week4', dataset, 'test',
                                   colorspace='gray', annotated=True)

    # Stabilize sequences
    train_stab, train_mask = video_stabilization.ngiaho_stabilization(train,
                                    gt_t, out_path, compareOutput, maxWidth)
    test_stab, test_mask = video_stabilization.ngiaho_stabilization(test,gt,
                                    out_path, compareOutput, maxWidth)

    # Add axis
    test_stab = test_stab[...,np.newaxis]
    train_stab = train_stab[...,np.newaxis]

    # Adaptive model prediction
    model = bg_subtraction.create_model(train)
    model_stab = bg_subtraction.create_model_mask(train_stab,
                                                  train_mask[1,:])

    # Compute AUC for Precision-Recall curves
    summaries_pred = []
    summaries_stab = []
    summaries_morph = []
    summaries_stab_morph = []

    for alpha in alpha_values:
        print(f"{alpha:0.4f}, ", end='', flush=True)
        #rho =rho[dataset]
        pred = bg_subtraction.predict(test, model, alpha, rho =rho[dataset])
        pred_stab = bg_subtraction.predict_masked(test_stab, test_mask[1,:],
                                model_stab, alpha, rho =rho[dataset])
        filled8 = morphology.imfill(pred, neighb=8)
        filled8_stab = morphology.imfill(pred_stab, neighb=8)
        clean = morphology.filter_small(filled8, bsize, neighb=4)
        clean_stab = morphology.filter_small(filled8_stab, bsize, neighb=4)

        #CLOSING
        st_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            se_close)
        clean = morphology.filter_morph(clean, cv2.MORPH_CLOSE,
                                              st_elem)
        clean_stab = morphology.filter_morph(clean_stab, cv2.MORPH_CLOSE,
                                              st_elem)

        #OPENING
        if (dataset == 'traffic'):
            st_elem = se_open
        else:
            st_elem = cv2.getStructuringElement(cv2.MORPH_RECT, se_open)

        morph = morphology.filter_morph(clean, cv2.MORPH_OPEN,
                                        st_elem)
        morph_stab = morphology.filter_morph(clean_stab, cv2.MORPH_OPEN,
                                        st_elem)

        summary_pred = metrics.eval_from_mask(pred, gt[:,0], gt[:,1])
        summary_stab = metrics.eval_from_mask(pred_stab,
                                              test_mask[0,:], test_mask[1,:])
        summary_morph = metrics.eval_from_mask(morph, gt[:,0], gt[:,1])
        summary_morph_stab = metrics.eval_from_mask(morph_stab,
                                              test_mask[0,:],test_mask[1,:])

        summaries_pred.append(summary_pred)
        summaries_stab.append(summary_stab)
        summaries_stab_morph.append(summary_morph_stab)
        summaries_morph.append(summary_morph)

    # AUC results
    auc_pred = metrics.auc(dict(data=summaries_pred), 'prec-rec')
    auc_stab = metrics.auc(dict(data=summaries_stab), 'prec-rec')
    auc_morph_stab = metrics.auc(dict(data=summaries_stab_morph), 'prec-rec')
    auc_morph = metrics.auc(dict(data=summaries_morph), 'prec-rec')
    print("\n")
    print("AUC for different tasks")
    print(f"AUC no imfill: {auc_pred:0.4f}")
    print(f"AUC no imfill stabilized: {auc_stab:0.4f}")
    print(f"AUC Morphology stabilized: {auc_morph_stab:0.4f}")
    print(f"AUC Morphology: {auc_morph:0.4f}")

    # More details for morphology
    alpha1, prec1, alpha2, prec2 = metrics.find_extrema(
        summaries_stab_morph, alpha_values, metrics.prec)
    print("")
    print("Summaries for Morphology")
    print(f"Precision min: {prec1:0.3f} (alpha {alpha1:0.3f})")
    print(f"Precision max: {prec2:0.3f} (alpha {alpha2:0.3f})")

    alpha1, rec1, alpha2, rec2 = metrics.find_extrema(
        summaries_stab_morph, alpha_values, metrics.recall)
    print(f"Recall min: {rec1:0.3f} (alpha {alpha1:0.3f})")
    print(f"Recall max: {rec2:0.3f} (alpha {alpha2:0.3f})")

    alpha1, fsco1, alpha2, fsco2 = metrics.find_extrema(
        summaries_stab_morph, alpha_values, lambda x: metrics.f_score(x,
                                                                beta=1))
    print(f"F1-score min: {fsco1:0.3f} (alpha {alpha1:0.3f})")
    print(f"F1-score max: {fsco2:0.3f} (alpha {alpha2:0.3f})")

    # Plot PR-curves for pred / fill4 / fill8 on the same plot
    plot_data = [dict(title=f"no imfill (AUC={auc_pred:0.4f})",
                      data=summaries_pred),
                 dict(title=f"no imfill stabilized (AUC={auc_stab:0.4f})",
                      data=summaries_stab),
                 dict(title=f"morphology stabilized (AUC={auc_morph_stab:0.4f})",
                      data=summaries_stab_morph),
                 dict(title=f"morphology: (AUC={auc_morph:0.4f})",
                      data=summaries_morph),
                 ]
    plt.figure()
    metrics.plot_precision_recall_curves(plot_data, same_plot=True)

    pred = bg_subtraction.predict(test, model, alpha2,
                                  rho=rho[dataset])
    pred_stab = bg_subtraction.predict_masked(test_stab, test_mask[1],
                                        model_stab, alpha2, rho=rho[dataset])


    filled8 = morphology.imfill(pred, neighb=8)
    filled8_stab = morphology.imfill(pred_stab, neighb=8)
    clean = morphology.filter_small(filled8, bsize, neighb=4)
    clean_stab = morphology.filter_small(filled8_stab, bsize, neighb=4)

    # CLOSING
    st_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        se_close)
    clean = morphology.filter_morph(clean, cv2.MORPH_CLOSE,
                                    st_elem)
    clean_stab = morphology.filter_morph(clean_stab, cv2.MORPH_CLOSE,
                                         st_elem)

    # OPENING
    if (dataset == 'traffic'):
        st_elem = se_open
    else:
        st_elem = cv2.getStructuringElement(cv2.MORPH_RECT, se_open)

    morph = morphology.filter_morph(clean, cv2.MORPH_OPEN,
                                    st_elem)
    morph_stab = morphology.filter_morph(clean_stab, cv2.MORPH_OPEN,
                                         st_elem)

    # Save individual gifs and an extra gif which compare them
    animations.video_recorder(pred, '', f"{dataset}_orig")
    animations.video_recorder(filled8, '', f"{dataset}_filled8")
    animations.video_recorder(clean, '', f"{dataset}_clean")
    animations.video_recorder(pred_stab, '', f"{dataset}_orig_stab")
    animations.video_recorder(morph_stab, '', f"{dataset}_morph_stab")
    animations.video_recorder(morph, '', f"{dataset}_morph")
    animations.video_recorder(test_mask[1,:], '', f"{dataset}_valid")

    #animations.save_comparison_gif(f"{dataset}_summary.gif", pred, pred_stab,
    #                               morph_stab)



    # animations.video_recorder(test_stab, '', f"{dataset}_stabilized")

    print("Done")