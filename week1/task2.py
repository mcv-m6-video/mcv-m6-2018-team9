import os

from evaluation import metrics


def run():

    tests_folder = os.path.join('datasets', 'results_testAB_changedetection',
                                'results', 'highway')
    gt_folder = os.path.join('datasets', 'results_testAB_changedetection',
                             'gt', 'groundtruth')

    test_a = metrics.eval_test_by_frame(tests_folder, gt_folder,
                                        test_prefix='test_A_', gt_prefix='gt',
                                        test_format='png', gt_format='png',
                                        exigence=2)

    test_b = metrics.eval_test_by_frame(tests_folder, gt_folder,
                                        test_prefix='test_B_', gt_prefix='gt',
                                        test_format='png', gt_format='png',
                                        exigence=2)

    test_a_accum = metrics.eval_test_history(tests_folder, gt_folder,
                                             test_prefix='test_A_',
                                             gt_prefix='gt',
                                             test_format='png',
                                             gt_format='png',
                                             exigence=2)

    test_b_accum = metrics.eval_test_history(tests_folder, gt_folder,
                                             test_prefix='test_B_',
                                             gt_prefix='gt',
                                             test_format='png',
                                             gt_format='png',
                                             exigence=2)

    tests = []
    tests.append(dict(title='Test A',  data=test_a))
    tests.append(dict(title='Test B',  data=test_b))

    metrics.plot_metric_history(tests, mode='TP_and_fg')
    metrics.plot_metric_history(tests, mode='F1')

    input('Press key to see cumulative results')

    tests_accum = []
    tests_accum.append(dict(title='Test A',  data=test_a_accum))
    tests_accum.append(dict(title='Test B',  data=test_b_accum))

    metrics.plot_metric_history(tests_accum, mode='TP_and_fg')
    metrics.plot_metric_history(tests_accum, mode='F1')
