import argparse

import week1
import week2
import week3
import week4
import week5


def main():
    parser = argparse.ArgumentParser(description='Run the tasks of each week.')
    parser.add_argument('week', help='Select a week')
    parser.add_argument('-t', '--task', metavar='task', type=int, default=1,
                        help='Select a task')
    parser.add_argument('-d', '--dataset', metavar='dataset', default='highway',
                        help='Select a task')
    args = parser.parse_args()

    # Week 1
    if args.week == 'week1':
        print('>>>> Executing: Week 1 - Task 1...')
        week1.task1.run()
        print('>>>> Executing: Week 1 - Task 2...')
        week1.task2.run()
        print('>>>> Executing: Week 1 - Task 3...')
        week1.task3.run()
        print('>>>> Executing: Week 1 - Task 4...')
        week1.task4.run()
        print('>>>> Executing: Week 1 - Task 5...')
        week1.task5.run()

    # Week 2
    elif args.week == 'week2_t1':
        week2.task1.run()
    elif args.week == 'week2_t2':
        week2.task2_curves.run(args.dataset)
    elif args.week == 'week2_t3':
        week2.task3.run()
    elif args.week == 'week2_t4':
        week2.task4.run()

    # Week 3
    elif args.week == 'week3':
        if args.task == 1:
            week3.task1_auc.run(args.dataset)
        elif args.task == 2:
            week3.task2.run()
        elif args.task == 3:
            week3.task3.run(args.dataset)
        elif args.task == 4:
            week3.task4.run(args.dataset)
        elif args.task == 5:
            week3.task5.run(args.dataset)

    # Week 4
    elif args.week == 'week4':
        if args.task == 1:
            week4.task1_gridsearch.run()
            week4.task1_qualitative.run()
            week4.task1_2.run()
        if args.task == 2:
            week4.task2_1.run()
            week4.task2_ngiaho.run('traffic')

    # Week 5
    elif args.week == 'week5':
        if args.task == 1:
            week5.task1_tracking.run(args.dataset)
            week5.task1_meanshift.run(args.dataset)
        elif args.task == 2:
            week5.task1_debug_speed.run(args.dataset)
        elif args.task == 3:
            week5.task3_tracking_app.run(args.dataset)


if __name__ == '__main__':
    main()
