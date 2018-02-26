import glob
import os
from copy import deepcopy
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from PIL import Image
np.set_printoptions(threshold=np.nan)



ALPHA = 10 ** (-12)


def eval_test(test_path, gt_path, test_prefix='', gt_prefix='',
              test_format='png', gt_format='png', exigence=2, desync=0):
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

        f_id = filename.replace(os.path.join(test_path, test_prefix), '')
        f_id = f_id.replace('.' + test_format, '')
        try:
            f_id = str(int(f_id) + desync).zfill(6)
        except:
            print('Erroneous type of Id in data files will result in fake '
                  'results.')
        filename_gt = os.path.join(gt_path, gt_prefix + f_id + '.' + gt_format)
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


def eval_test_by_frame(test_path, gt_path, test_prefix='', gt_prefix='',
                       test_format='png', gt_format='png', exigence=2):
    """
    Evaluates some test results against a given ground truth frame by frame

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

    :return: (list of dict) results of the test analysis at each frame. Each
    frame contains the following keys:

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

    data_history = []

    data = dict(TP=0, FP=0, FN=0, TN=0)

    for filename in glob.glob(os.path.join(test_path,
                                           test_prefix + '*.' + test_format)):
        pil_img_test = Image.open(filename)
        img_test = np.array(pil_img_test)

        f_id = filename.replace(os.path.join(test_path, test_prefix), '')
        f_id = f_id.replace('.' + test_format, '')
        filename_gt = os.path.join(gt_path, gt_prefix + f_id + '.' + gt_format)
        pil_img_gt = Image.open(filename_gt)
        real_img_gt = np.array(pil_img_gt)
        img_gt = np.where(real_img_gt > fg_thresh, 1, 0)

        trues_test = img_test.astype(bool)
        trues_gt = img_gt.astype(bool)
        img_tp = np.logical_and(trues_test, trues_gt)
        img_fp = np.logical_and(trues_test, np.logical_not(trues_gt))
        img_fn = np.logical_and(np.logical_not(trues_test), trues_gt)
        img_tn = np.logical_not(np.logical_and(trues_test, trues_gt))

        data['TP'] = img_tp.sum()
        data['FP'] = img_fp.sum()
        data['FN'] = img_fn.sum()
        data['TN'] = img_tn.sum()

        data_history.append(deepcopy(data))

    return data_history


def eval_test_history(test_path, gt_path, test_prefix='', gt_prefix='',
                      test_format='png', gt_format='png', exigence=2):
    """
    Evaluates some test results evolution against a given ground truth

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

    :return: (dict) results of the test analysis until each frame. At each frame
    it has the keys:

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

    data_history = []

    accum_data = dict(TP=0, FP=0, FN=0, TN=0)

    for filename in glob.glob(os.path.join(test_path,
                                           test_prefix + '*.' + test_format)):
        pil_img_test = Image.open(filename)
        img_test = np.array(pil_img_test)

        f_id = filename.replace(os.path.join(test_path, test_prefix), '')
        f_id = f_id.replace('.' + test_format, '')
        filename_gt = os.path.join(gt_path, gt_prefix + f_id + '.' + gt_format)
        pil_img_gt = Image.open(filename_gt)
        real_img_gt = np.array(pil_img_gt)
        img_gt = np.where(real_img_gt > fg_thresh, 1, 0)

        trues_test = img_test.astype(bool)
        trues_gt = img_gt.astype(bool)
        img_tp = np.logical_and(trues_test, trues_gt)
        img_fp = np.logical_and(trues_test, np.logical_not(trues_gt))
        img_fn = np.logical_and(np.logical_not(trues_test), trues_gt)
        img_tn = np.logical_not(np.logical_and(trues_test, trues_gt))

        accum_data['TP'] += img_tp.sum()
        accum_data['FP'] += img_fp.sum()
        accum_data['FN'] += img_fn.sum()
        accum_data['TN'] += img_tn.sum()

        data_history.append(deepcopy(accum_data))

    return data_history


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
    return data['TP'] / (data['TP']+data['FP'] + ALPHA)


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
    return data['TP'] / (data['TP'] + data['FN'] + ALPHA)


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
                              ((beta**2) * (prec(data)) + recall(data) + ALPHA))


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
        print('Precison: ' + str(prec(test['data'])))
        print('Recall: ' + str(recall(test['data'])))
        print('F1-score: ' + str(f1_score(test['data'])))
        print('\n' * 2)


def plot_metric_history(test_historicals, mode):
    """
    Plots some list of sets of test results by frame. The data to display will
    depend on the mode.

    :param test_historicals: (list of list of dicts) collection of test
    historicals where each historical contains a set (list) of tests consisting
    on the data dictionary returned by eval_test.
    :param mode: (str) Options:
        - 'TP_and_fg': for each test set will display its TP and total
        foreground by test.
        - 'F1 score': for each test set will display its F1 score.
        - Otherwise it will do nothing.
    :return:
    """
    styles = [('c-', 'b-', 'cyan', 'blue'), ('m-', 'r-', 'magenta', 'red'),
              ('k-', 'g-', 'black', 'green'), ('w-', 'y-', 'white', 'yellow')]
    if mode is 'TP_and_fg':
        plt.title('TP and total pixels as foreground')
        patches = []
        for i, test_h in enumerate(test_historicals):
            plt.plot(range(1, len(test_h['data'])+1),
                     [data['TP'] for data in test_h['data']],
                     styles[i % 4][0])
            plt.plot(range(1, len(test_h['data'])+1),
                     [(data['TP'] + data['FP']) for data in test_h['data']],
                     styles[i % 4][1])
            patch_1 = mpatches.Patch(color=styles[i % 4][2],
                                     label=test_h['title'] + ' TP')
            patch_2 = mpatches.Patch(color=styles[i % 4][3],
                                     label=test_h['title'] +
                                     ' total foreground')
            patches.append(patch_1)
            patches.append(patch_2)

        plt.legend(handles=patches)
        plt.show()
    elif mode is 'F1':
        plt.title('F1 score')
        patches = []
        for i, test_h in enumerate(test_historicals):
            plt.plot(range(1, len(test_h['data'])+1),
                     [f1_score(data) for data in test_h['data']],
                     styles[i % 4][1])
            patch = mpatches.Patch(color=styles[i % 4][3],
                                   label=test_h['title'])
            patches.append(patch)

        plt.legend(handles=patches)
        plt.show()
    else:
        print('Invalid option')


def plot_desynchronization_effect(tests_data, desynchronization_range=[0]):

    styles = [('b-', 'blue'), ('r-', 'red'), ('g-', 'green'), ('y-', 'yellow')]

    plt.title('F1 score')
    patches = []

    for i, desync_data in enumerate(tests_data):
        plt.plot(desynchronization_range,
                 [f1_score(data) for data in desync_data['data']],
                 styles[i % 4][0])
        patch = mpatches.Patch(color=styles[i % 4][1],
                               label=desync_data['title'])
        patches.append(patch)

    plt.legend(handles=patches)
    plt.show()


def msen(pred, gt):
    valid = gt[:,:,2] == 1
    gt_valid = gt[valid]
    pred_valid = pred[valid]

    vect_err = gt_valid[:,:2] - pred_valid[:,:2]
    squared_err = np.sum(vect_err**2, axis=1)
    err = np.sqrt(squared_err)
    hit = err < 3.0

    msen = np.mean(squared_err)
    pepn = 100 * (1 - np.mean(hit))

    print('Valid GT Vectors: ', hit.size)
    print('Valid Error Vectors (< 3): ', hit.sum())
    print('PEPN: ', pepn)
    print('MSEN: ', msen)

    plt.hist(vect_err, bins=25, normed=True)
    plt.xlabel('Error Magintude')
    plt.ylabel('% of Pixels')
    plt.title(" MSEN ")
    plt.show()

    return vect_err, msen, pepn
