import os

from evaluation import metrics, visual


def run():

    tests_folder = os.path.join('datasets', 'results_testAB_changedetection',
                                'results', 'highway')
    gt_folder = os.path.join('datasets', 'results_testAB_changedetection',
                             'gt', 'groundtruth')

    test_a = metrics.eval_test(tests_folder, gt_folder, test_prefix='test_A_',
                               gt_prefix='gt', test_format='png',
                               gt_format='png', exigence=2)

    test_b = metrics.eval_test(tests_folder, gt_folder, test_prefix='test_B_',
                               gt_prefix='gt', test_format='png',
                               gt_format='png', exigence=2)

    tests = []
    tests.append(dict(description='Test A (motion inside the region of interest'
                                  ' as foreground)', data=test_a))
    tests.append(dict(description='Test B (motion inside the region of interest'
                                  ' as foreground)', data=test_b))

    metrics.summarize_tests(tests)

    input('Press any key to continue to the qualitative results')

    visual.play_matches(tests_folder, 'test_A_*.png')
    visual.play_matches(tests_folder, 'test_B_*.png')


