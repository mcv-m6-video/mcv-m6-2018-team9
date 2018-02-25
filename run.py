import argparse

import week1


def main():
    parser = argparse.ArgumentParser(description='Run the tasks of each week.')
    parser.add_argument('week', choices=['week1', 'week2', 'week3', 'week4'],
                        help='Execute the tasks of a week')

    args = parser.parse_args()
    if args.week == 'week1':
        print('>>>> Executing: Week 1 - Task 1...')
        week1.task1.run()
        print('>>>> Executing: Week 1 - Task 2...')
        week1.task2.run()
        print('>>>> Executing: Week 1 - Task 5...')
        week1.task5.run()


if __name__ == '__main__':
    main()
