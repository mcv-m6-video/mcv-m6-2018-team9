import glob
import os
import numpy as np
from PIL import Image


def eval_test(test_path, gt_path, test_prefix='', gt_prefix='',
              test_format='png', gt_format='png', exigence=2):

    if exigence is 0:
        fg_thresh = 25
    elif exigence is 1:
        fg_thresh = 75
    elif exigence is 3:
        fg_thresh = 200
    else:
        fg_thresh = 100

    data = dict(TP=0, FP=0, FN=0, TN=0)

    for filename in glob.glob(os.path.join(test_path,
                                           test_prefix + '*.' + test_format)):
        pil_img_test = Image.open(filename)
        img_test = np.array(pil_img_test)

        id = filename.replace(os.path.join(test_path, test_prefix), '')
        id = id.replace('.' + test_format, '')
        filename_gt = os.path.join(gt_path, gt_prefix + id + '.' + gt_format)
        pil_img_gt = Image.open(filename_gt)
        real_img_gt = np.array(pil_img_gt)
        img_gt = np.where(real_img_gt > fg_thresh, 1, 0)

        trues_test = img_test.astype(bool)
        trues_gt = img_gt.astype(bool)
        img_tp = np.logical_and(trues_test, trues_gt)
        img_fp = np.logical_and(trues_test, np.logical_not(trues_gt))
        img_fn = np.logical_and(np.logical_not(trues_test), trues_gt)
        img_tn = np.logical_not(np.logical_and(trues_test, trues_gt))

        data['TP'] += img_tp.sum()
        data['FP'] += img_fp.sum()
        data['FN'] += img_fn.sum()
        data['TN'] += img_tn.sum()

    return data


def prec(data):
    return data['TP'] / (data['TP']+data['FP'])


def recall(data):
    return data['TP'] / (data['TP'] + data['FN'])


def f_score(data, beta=1):
    return (1 + (beta**2)) * ((prec(data) * recall(data)) /
                              ((beta**2) * (prec(data)) + recall(data)))


def f1_score(data):
    return f_score(data, beta=1)
