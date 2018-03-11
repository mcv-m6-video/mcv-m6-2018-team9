import numpy as np
import matplotlib.pyplot as plt

from data import cdnet
from video import bg_subtraction, morphology
from evaluation import metrics, animations


def run(dataset):
    # Model parameters
    rho = 0.20
    alpha_values = np.concatenate([np.linspace(0, 10, 30),
                                   np.linspace(11, 40, 10)])

    # Read dataset
    train = cdnet.read_sequence('week3', dataset, 'train', colorspace='gray',
                                annotated=False)
    test, gt = cdnet.read_sequence('week3', dataset, 'test',
                                   colorspace='gray', annotated=True)

    # Adaptive model prediction
    model = bg_subtraction.create_model(train)

    # Compute AUC for Precision-Recall curves (3 methods: pred, fill4, fill8)
    summaries_pred = []
    summaries_fill4 = []
    summaries_fill8 = []

    for alpha in alpha_values:
        print(f"{alpha:0.4f}, ", end='', flush=True)
        pred = bg_subtraction.predict(test, model, alpha, rho=rho)
        filled4 = morphology.imfill(pred, neighb=4)
        filled8 = morphology.imfill(pred, neighb=8)

        summary_pred = metrics.eval_from_mask(pred, gt[:,0], gt[:,1])
        summary_fill4 = metrics.eval_from_mask(filled4, gt[:,0], gt[:,1])
        summary_fill8 = metrics.eval_from_mask(filled8, gt[:,0], gt[:,1])

        summaries_pred.append(summary_pred)
        summaries_fill4.append(summary_fill4)
        summaries_fill8.append(summary_fill8)

    # AUC results for the 3 methods evaluated (pred, fill4, fill8)
    auc_pred = metrics.auc(dict(data=summaries_pred), 'prec-rec')
    auc_fill4 = metrics.auc(dict(data=summaries_fill4), 'prec-rec')
    auc_fill8 = metrics.auc(dict(data=summaries_fill8), 'prec-rec')
    print("\n")
    print(f"AUC no imfill: {auc_pred:0.4f}")
    print(f"AUC imfill 4-neighb: {auc_fill4:0.4f}")
    print(f"AUC imfill 8-neighb: {auc_fill8:0.4f}")

    # auc2_pred = metrics.auc2(summaries_pred, 'prec-rec')
    # auc2_fill4 = metrics.auc2(summaries_fill4, 'prec-rec')
    # auc2_fill8 = metrics.auc2(summaries_fill8, 'prec-rec')
    # print(f"AUC no imfill: {auc2_pred:0.4f}")
    # print(f"AUC imfill 4-neighb: {auc2_fill4:0.4f}")
    # print(f"AUC imfill 8-neighb: {auc2_fill8:0.4f}")

    # More details for imfill 8
    alpha1, prec1, alpha2, prec2 = metrics.find_extrema(
        summaries_fill8, alpha_values, metrics.prec)
    print("")
    print("Summaries for imfill 8")
    print(f"Precision min: {prec1:0.3f} (alpha {alpha1:0.3f})")
    print(f"Precision max: {prec2:0.3f} (alpha {alpha2:0.3f})")

    alpha1, rec1, alpha2, rec2 = metrics.find_extrema(
        summaries_fill8, alpha_values, metrics.recall)
    print(f"Recall min: {rec1:0.3f} (alpha {alpha1:0.3f})")
    print(f"Recall min: {rec2:0.3f} (alpha {alpha2:0.3f})")

    alpha1, fsco1, alpha2, fsco2 = metrics.find_extrema(
        summaries_fill8, alpha_values, lambda x: metrics.f_score(x, beta=2))
    print(f"F2-score min: {fsco1:0.3f} (alpha {alpha1:0.3f})")
    print(f"F2-score min: {fsco2:0.3f} (alpha {alpha2:0.3f})")

    # Plot PR-curves for pred / fill4 / fill8 on the same plot
    plot_data = [dict(title=f"no imfill (AUC={auc_pred:0.4f})",
                      data=summaries_pred),
                 dict(title=f"imfill (AUC={auc_fill4:0.4f})",
                      data=summaries_fill4),
                 dict(title=f"imfill (AUC={auc_fill8:0.4f})",
                      data=summaries_fill8)]
    metrics.plot_precision_recall_curves(plot_data, same_plot=True)
