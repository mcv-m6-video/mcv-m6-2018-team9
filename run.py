import argparse

import week1
import week2


def main():
    parser = argparse.ArgumentParser(description='Run the tasks of each week.')
    parser.add_argument('week', choices=['week1', 'week2', 'week3', 'week4','week5'],
                        help='Execute the tasks of a week')

    args = parser.parse_args()
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

    elif args.week == 'week2':
        print('>>>> Executing: Week 2 - Task 1...')
        week2.task1.run()
        print('>>>> Executing: Week 2 - Task 2...')
        #week2.task2.run()

if __name__ == '__main__':
    main()
