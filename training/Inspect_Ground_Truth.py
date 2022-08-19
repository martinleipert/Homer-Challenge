import fiftyone
from fiftyone import load_dataset
import argparse

# By Martin Leipert
# martin.leipert@fau.de
# 19.08.2022
# Open a fiftyone session of the selected dataset
# The data then can be inspected via a webbrowser


def main():

    # List all available datasets
    datasets = [
        "ICFHR2022_train",
        "ICFHR2022_artificial"
    ]

    # Let the user decide which set to open
    parser = argparse.ArgumentParser("select the dataset to display")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--originalTraining", action="store_true", type=bool,
                       help="The original training dataset provided by LME")
    group.add_argument("--syntheticTraining", action="store_true", type=bool,
                       help="The synthetic dataset created with google fonts, "
                            "some textures and ancient greek txt-Files and PIL")

    args = parser.parse_args()

    # Select the dataset
    if args.originalTraining is True:
        dataset = load_dataset(datasets[0])
    elif args.syntheticTraining is True:
        dataset = load_dataset(datasets[1])
    else:
        exit(1)

    # Launch browser session
    session = fiftyone.launch_app(dataset, port=5151)

    # Wait for user to open session, otherwise code would exit immediatly
    session.wait()


if __name__ == "__main__":
    main()
