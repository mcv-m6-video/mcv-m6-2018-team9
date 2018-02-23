import glob
import os
import numpy as np
from PIL import Image


def eval_test(test_path, gt_path, test_prefix='', gt_prefix='',
              test_format='png', gt_format='png', exigence=2):
    """
    Evaluates some test results against a given ground truth

    :param test_path: (str) relative or absolute path to the test results images
    :param gt_path: (str) relative or absolute path to the ground truth images
    :param test_prefix: (str) prefix of the test files before their ID (e.g.
    test_A_001235.png has test_A_ as prefix)
    :param gt_prefix: (str) prefix of the ground truth files before their ID
    (e.g. gt001235.png has gt as prefix)
    :param test_format: (str) format of the test images
    :param gt_format: (str) format of the ground truth images
    :param exigence: (int) tells how easy will be from a pixel to be foreground
    in the ground truth:

        - 0: all non-static pixels will be taken as foreground
        - 1: all non-static pixels excepting hard shadows will be taken as
        foreground
        - 2: only pixels with motion inside the region of interest will be taken
        as foreground
        - 3: only pixels with known motion inside the region of interest will be
        taken as foreground
        - Else exigence=2 will be assumed

    :return: (dict) results of the test analysis.

        - TP: (int) true positives
        - FP: (int) false positives
        - FN: (int) false negatives
        - TN: (int) true negatives

    """

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
    """
    Precision

    :param (dict) results of the test analysis.

        - TP: (int) true positives
        - FP: (int) false positives
        - FN: (int) false negatives
        - TN: (int) true negatives

    :return: (int)
    """
    return data['TP'] / (data['TP']+data['FP'])


def recall(data):
    """
    Recall

    :param (dict) results of the test analysis.

        - TP: (int) true positives
        - FP: (int) false positives
        - FN: (int) false negatives
        - TN: (int) true negatives

    :return: (int)
    """
    return data['TP'] / (data['TP'] + data['FN'])


def f_score(data, beta=1):
    """
    F_beta score

    :param (dict) results of the test analysis.

        - TP: (int) true positives
        - FP: (int) false positives
        - FN: (int) false negatives
        - TN: (int) true negatives

    :return: (int)
    """
    return (1 + (beta**2)) * ((prec(data) * recall(data)) /
                              ((beta**2) * (prec(data)) + recall(data)))


def f1_score(data):
    """
    F_1 score

    :param (dict) results of the test analysis.

        - TP: (int) true positives
        - FP: (int) false positives
        - FN: (int) false negatives
        - TN: (int) true negatives

    :return: (int)
    """
    return f_score(data, beta=1)


def summarize_tests(tests):
    """
    Prints results from some tests

    :param tests: (list of dict) contains a set of analyzed tests where each
    test has the following keys:

        - description: (str) descriptive string of the test
        - data: (dict) results of the test analysis.

            - TP: (int) true positives
            - FP: (int) false positives
            - FN: (int) false negatives
            - TN: (int) true negatives
    """
    for test in tests:
        print('-' * 100)
        print(test['description'])
        print('-' * 100)
        print('TP: ' + str(test['data']['TP']))
        print('FP: ' + str(test['data']['FP']))
        print('FN: ' + str(test['data']['FN']))
        print('TN: ' + str(test['data']['TN']))
        print('\n' * 2)

