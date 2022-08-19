from training_func import training_func

DATASETS_AVAILABLE = ["ICFHR2022_train", "ICFHR2022_artificial"]
epochs = 50

PATH = "model_pretraining.pth"
BATCH_SIZE = 1


def main():
    dataset_name_artificial = DATASETS_AVAILABLE[1]
    dataset_name_training = DATASETS_AVAILABLE[0]

    # First training is pretraining
    # epochs = 50
    training_func(dataset_name=dataset_name_artificial, model_dir="model_store/artificial",
                  model_name="model_artificial.pth", epochs=1)

    # The model trained on real papyri
    # epochs=200,
    training_func(dataset_name=dataset_name_training, model_dir="model_store/papyri",
                  model_name="model_artificial.pth", epochs=1,
                  initialization="model_store/artificial/model_artificial.pth")


if __name__ == "__main__":
    main()
