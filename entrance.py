from pddl_handler import *

if __name__ == '__main__':
    try:
        model_builder.main()
        print("Done.")
    except Exception as e:
        print("Program failed caused by some reason. Please check the log file for more details.")

