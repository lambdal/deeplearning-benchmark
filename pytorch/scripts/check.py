import os
import argparse
from termcolor import colored

def main():
    parser = argparse.ArgumentParser(description='Gather benchmark results.')

    parser.add_argument('--path', type=str, default='results/48GB',
                        help='path that has the results of all tests')
    args = parser.parse_args()

    print("Check results folder : {}".format(args.path))

    if os.path.exists(args.path):
        for taskname in os.listdir(args.path):
            # Get the txt file inside of the folder 
            if not taskname.endswith(".txt"):
                task_dir = os.path.join(args.path, taskname)
                for filename in os.listdir(task_dir):
                    if filename.endswith(".txt"):
                        with open(os.path.join(task_dir, filename), 'r') as f:
                            last_line = f.readlines()[-1]
                            if "DONE!" in last_line:
                                print(colored("{: <40} : {: >20}".format(taskname, "sucessful"), "green"))
                            else:
                                print(colored("{: <40} : {: >20}".format(taskname, "unsucessful"), "red"))


if __name__ == "__main__":
    main()