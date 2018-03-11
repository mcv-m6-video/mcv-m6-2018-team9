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
import sklearn.metrics as skmetrics


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


def eval_from_mask(result_mask, gt_mask, valid_mask=None):
    """
    Evaluates some test results evolution against a given ground truth from
    logical matrices.
    :param result_mask: (numpy array) 3D matrix (2D + temporal dimension) with
    the mask of the results for each one, where 0 is background and 1 is
    foreground.
    :param gt_mask: (numpy array) 3D matrix (2D + temporal dimension) with
    the mask of the ground truth for each one, where 0 is background and 1 is
    foreground.
    :return: (dict) results of the test analysis.
        - TP: (int) true positives
        - FP: (int) false positives
        - FN: (int) false negatives
        - TN: (int) true negatives
    """
    if valid_mask is None:
        valid_mask = np.ones(gt_mask.shape, dtype='bool')

    tp = result_mask & gt_mask & valid_mask
    fp = result_mask & ~gt_mask & valid_mask
    fn = ~result_mask & gt_mask & valid_mask
    tn = ~result_mask & ~gt_mask & valid_mask

    return dict(TP=tp.sum(), FP=fp.sum(), FN=fn.sum(), TN=tn.sum())


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


def find_extrema(summaries, thresholds, metric):
    """Find min and max metric values for a given list of summaries

    Each summary is a dictionary with TP/FP/FN/TN values, after invoking the
    `eval_from_mask` function. These summaries must be obtained evaluating a
    given model with different thresholds, which are passed to the function
    through the `thresholds` parameter.

    Args:
      summaries: (list of dict) list of summaries obtained evaluating the
        output of a model with `eval_from_mask`.
      thresholds: (list of floats) the corresponding thresholds used in the
        summaries. Typically thresholds are alpha and rho in bg subtraction
        models.
      metric: (str or func) The metric to which find the extrema. Can be
        'precision', 'recall' or any custom python function.

    Returns:
      A tuple (thresh_min, metric_min, thresh_max, metric_max), where
      `thresh_min` is the threshold value that has the minimum with respect to
      `metric` and `metric_min` is that value. Analogously for the max value.

    """
    if metric == 'precision':
        values = [prec(s) for s in summaries]
        argmax = np.argmax(values)
        argmin = np.argmin(values)
    elif metric == 'recall':
        values = [prec(s) for s in summaries]
        argmax = np.argmax(values)
        argmin = np.argmin(values)
    else:
        values = [metric(s) for s in summaries]
        argmax = np.argmax(values)
        argmin = np.argmin(values)

    return (thresholds[argmin], values[argmin],
            thresholds[argmax], values[argmax])


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


def plot_results_by_some_param(tests, param_range, param_name, mode):
    """
    Plots some test results by some parameter (e.g. some threshold). The data to
    display will depend on the mode.

    :param tests: (list of dicts) list of tests where each test has the
    following keys:

        - title: (str) descriptive string of the test
        - data: (list of dict) results of the test analysis returned by
        eval_test

    :param param_range: (list of int) range of values for the parameter
    :param param_name: (string) name for the parameter
    :param mode: (str) Options:
        - 'main': will display TP, FP, FN and TN.
        - 'prec-rec': will display precision and recall.
        - 'F1': will display F1 score.
        - 'prec-rec-f1': will display precision, recall and F1 score together.
        - Otherwise it will do nothing.

    NOTE: Colors will repeat for more than 2 tests.
    """
    styles = [('c-', 'b-', 'cyan', 'blue'), ('m-', 'r-', 'magenta', 'red'),
              ('k-', 'g-', 'black', 'green'), ('ko-', 'y-', 'white', 'yellow')]
    if mode is 'main':
        plt.title('TP, FP, FN and TN by ' + param_name)
        patches = []
        for i, test in enumerate(tests):
            try:
                assert len(test['data']) is len(param_range)
            except:
                raise Exception(
                    'The length of test data should be as long as the range of '
                    + param_name)
            plt.plot(param_range, [data['TP'] for data in test['data']],
                     styles[0][i % 2])
            plt.plot(param_range, [data['FP'] for data in test['data']],
                     styles[1][i % 2])
            plt.plot(param_range, [data['FN'] for data in test['data']],
                     styles[2][i % 2])
            plt.plot(param_range, [data['TN'] for data in test['data']],
                     styles[3][i % 2])
            patch_1 = mpatches.Patch(color=styles[0][2 + (i % 2)],
                                     label=test['title'] + ' TP')
            patch_2 = mpatches.Patch(color=styles[1][2 + (i % 2)],
                                     label=test['title'] + ' FP')
            patch_3 = mpatches.Patch(color=styles[2][2 + (i % 2)],
                                     label=test['title'] + ' FN')
            patch_4 = mpatches.Patch(color=styles[3][2 + (i % 2)],
                                     label=test['title'] + ' TN')
            patches.append(patch_1)
            patches.append(patch_2)
            patches.append(patch_3)
            patches.append(patch_4)

        plt.legend(handles=patches)
        plt.xlabel(param_name)
        plt.show()
    elif mode == 'prec-rec':
        plt.title('Precision & Recall by ' + param_name)
        patches = []
        for i, test in enumerate(tests):
            try:
                assert len(test['data']) is len(param_range)
            except:
                raise Exception(
                    'The length of test data should be as long as the range of '
                    + param_name)
            plt.plot(param_range, [prec(data) for data in test['data']],
                     styles[0][i % 2])
            plt.plot(param_range, [recall(data) for data in test['data']],
                     styles[1][i % 2])
            patch_1 = mpatches.Patch(color=styles[0][2 + (i % 2)],
                                     label=test['title'] + ' Precision')
            patch_2 = mpatches.Patch(color=styles[1][2 + (i % 2)],
                                     label=test['title'] + ' Recall')
            patches.append(patch_1)
            patches.append(patch_2)

        plt.legend(handles=patches)
        plt.xlabel(param_name)
        plt.show()
    elif mode is 'F1':
        plt.title('F1 score by ' + param_name)
        patches = []
        for i, test in enumerate(tests):
            try:
                assert len(test['data']) is len(param_range)
            except:
                raise Exception(
                    'The length of test data should be as long as the range of '
                    + param_name)
            plt.plot(param_range, [f1_score(data) for data in test['data']],
                     styles[0][i % 2])
            patch = mpatches.Patch(color=styles[0][2 + (i % 2)],
                                   label=test['title'])
            patches.append(patch)

        plt.legend(handles=patches)
        plt.xlabel(param_name)
        plt.show()
    elif mode == 'prec-rec-f1':
        plt.title('Precision, Recall & F1 score by ' + param_name)
        patches = []
        for i, test in enumerate(tests):
            try:
                assert len(test['data']) is len(param_range)
            except:
                raise Exception(
                    'The length of test data should be as long as the range of '
                    + param_name)
            plt.plot(param_range, [prec(data) for data in test['data']],
                     styles[0][i % 2])
            plt.plot(param_range, [recall(data) for data in test['data']],
                     styles[1][i % 2])
            plt.plot(param_range, [f1_score(data) for data in test['data']],
                     styles[2][i % 2])
            patch_1 = mpatches.Patch(color=styles[0][2 + (i % 2)],
                                     label=test['title'] + ' Precision')
            patch_2 = mpatches.Patch(color=styles[1][2 + (i % 2)],
                                     label=test['title'] + ' Recall')
            patch_3 = mpatches.Patch(color=styles[2][2 + (i % 2)],
                                     label=test['title'] + ' F1 score')
            patches.append(patch_1)
            patches.append(patch_2)
            patches.append(patch_3)

        plt.legend(handles=patches)
        plt.xlabel(param_name)
        plt.show()
    else:
        print('Invalid option')


def plot_precision_recall_curves(tests, same_plot=False):
    """Plots the precision-recall curve for a set of tests.

    :param tests: (list of dicts) list of tests where each test has the
    following keys:
       - title: (str) descriptive string of the test
        - data: (list of dict) results of the test analysis returned by
        eval_test
    :param same_plot: (bool) when True, all the curves are plot in the same
      figure.

    NOTE: Colors will repeat for more than 4 tests.

    """
    styles = [('c-', 'cyan'), ('m-', 'magenta'), ('g-', 'green'),
              ('y-', 'yellow')]

    plt.title('Precision-Recall curve')
    patches = []
    for i, test in enumerate(tests):
        precisions = [prec(data) for data in test['data']]
        recalls = [recall(data) for data in test['data']]
        plt.plot(recalls, precisions, styles[i % 4][0])
        patch = mpatches.Patch(color=styles[i % 4][1], label=test['title'])
        patches.append(patch)

        if not same_plot:
            plt.legend(handles=patches)
            plt.ylabel('Precision')
            plt.xlabel('Recall')
            plt.show()

    if same_plot:
        plt.legend(handles=patches)
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.show()


def plot_auc_by_removed_area(tests, p_range, alpha_range):
    styles = [('c-', 'cyan'), ('m-', 'magenta'), ('g-', 'green'),
              ('y-', 'yellow')]
    patches = []
    for i, test in enumerate(tests):
        try:
            assert len(test['data']) is len(p_range)
        except:
            raise Exception(
                'The length of test data should be as long as the range of P')
        aucs = [auc(dict(title=test['title'], data=data),
                               'prec-rec', alpha_range)
                           for data in test['data']]
        plt.plot(p_range, aucs, styles[i % 4][0])
        lab = test['title'] + '(max: ' + '{:.2f}'.format(max(aucs)) + ' at P=' \
              + str(p_range[aucs.index(max(aucs))]) + ')'
        patch = mpatches.Patch(color=styles[i % 4][1], label=lab)
        patches.append(patch)

    plt.legend(handles=patches)
    plt.ylabel('AUC')
    plt.xlabel('P')
    plt.show()


def auc(test, curve_type, thr_range=[]):
    """
    Returns the area under the curve for some measures from a test.

    :param test:  (dict) dictinary with the following keys:

        - title: (str) descriptive string of the test
        - data: (list of dict) results of the test analysis returned by
        eval_test

    :param curve_type: (str) Options:

        - 'prec-rec': will compute the AUC of the precision-recall curve.
        - 'F1': will compute the AUC of the F1-score vs threshold curve.
        In this option it is necessary to specify the range of thresholds.

    :param thr_range: (list of int) range of values for the threshold. Only used
    in the F1-score option
    :return: (int) Area Under the Curve for the specified mode and test
    """
    if curve_type == 'prec-rec':
        random_precisions = [prec(data) for data in test['data']]
        random_recalls = [recall(data) for data in test['data']]
        recalls = [y for y,_ in sorted(zip(random_recalls, random_precisions))]
        precisions = [x for _,x in
                      sorted(zip(random_recalls, random_precisions))]
        recalls = [0] + recalls
        precisions = [precisions[0]] + precisions
        lower_precisions = [min(precisions[i], precisions[i + 1]) for i in
                            range(len(precisions) - 1)]
        return sum(np.array(lower_precisions) * abs(np.diff(recalls))) \
               + sum(abs(np.diff(precisions)) * abs(np.diff(recalls))) / 2
    elif curve_type is 'F1':
        try:
            assert len(test['data']) is len(thr_range)
        except:
            raise Exception('The number of tests should be as long as the '
                            'range of thresholds')
        f1scores = [f1_score(data) for data in test['data']]
        lower_f1 = [min(f1scores[i], f1scores[i + 1]) for i in
                    range(len(f1scores) - 1)]
        return sum(np.array(lower_f1) * abs(np.diff(thr_range))) \
               + sum(abs(np.diff(f1scores)) * abs(np.diff(thr_range))) / 2
    else:
        print('Invalid option')
        return None


def auc2(summaries, curve_type):
    """Alternative implementation for AUC computation using scikit-learn API"""
    if curve_type == 'prec-rec':
        precisions = [prec(s) for s in summaries]
        recalls = [recall(s) for s in summaries]
        return skmetrics.auc(recalls, precisions, reorder=True)
    else:
        raise ValueError("Unknown curve_type")


def msen(pred, gt):
    """
        msen

        :param  numpy arrays with shape [height, width, 3]. The first and second channels
                denote the corresponding optical flow 2D vector (u, v). The third channel
                is a mask denoting if an optical flow 2D vector exists for that pixel.
                Vector components u and v values range [-512..512].
                - pred: test resulting optical flow matrix
                - gt: ground truth optical flow matrix

        :return: - (float)msen: Mean Square Error in Non-Occluded Areas
                 - (float) pepn: Percentage of Error Pixels in Non-Occluded Areas
                  - numpy array with shape [height,width]) img_err: magnitude of the squared error
                  between input images on each position of the matrix.
                  - numpy array with shape [height,1] err: vector containing error motion vectors magnitude

        """
    valid = gt[:,:,2] == 1
    gt_valid = gt[valid]
    pred_valid = pred[valid]
    img_err = np.zeros(shape=gt[:,:,1].shape)
    err = gt_valid[:,:2] - pred_valid[:,:2]
    squared_err = np.sum(err**2, axis=1)
    vect_err = np.sqrt(squared_err)
    hit = vect_err < 3.0

    print('Valid GT Vectors: ', hit.size)
    print('Valid Error Vectors (< 3): ', hit.sum())

    img_err[valid] = vect_err
    msen = np.mean(vect_err)
    pepn = 100 * (1 - np.mean(hit))

    return msen, pepn, img_err, vect_err
