import os

from evaluation import metrics


def run():

    desynchronization_range = range(-25, 25)

    tests_folder = os.path.join('datasets', 'results_testAB_changedetection',
                                'results', 'highway')
    gt_folder = os.path.join('datasets', 'results_testAB_changedetection',
                             'gt', 'groundtruth')

    test_a = [metrics.eval_test(tests_folder, gt_folder,
                                test_prefix='test_A_', gt_prefix='gt',
                                test_format='png', gt_format='png',
                                exigence=2, desync=desync)
              for desync in desynchronization_range]

    test_b = [metrics.eval_test(tests_folder, gt_folder,
                                test_prefix='test_B_', gt_prefix='gt',
                                test_format='png', gt_format='png',
                                exigence=2, desync=desync)
              for desync in desynchronization_range]

    tests = []
    tests.append(dict(title='Test A',  data=test_a))
    tests.append(dict(title='Test B',  data=test_b))

    metrics.plot_desynchronization_effect(tests, desynchronization_range)
