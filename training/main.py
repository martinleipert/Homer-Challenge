from training_func import training_func
import torch

# Speed up
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

DATASETS_AVAILABLE = ["ICFHR2022_train", "ICFHR2022_artificial"]
epochs = 50

PATH = "model_pretraining.pth"
BATCH_SIZE = 1


def main():
    dataset_name_artificial = DATASETS_AVAILABLE[1]
    dataset_name_training = DATASETS_AVAILABLE[0]

    # First training is pretraining
    training_func(dataset_name=dataset_name_artificial, model_dir="model_store/artificial",
                  model_name="model_artificial.pth", epochs=120, decay=40, learning_rate=1e-2, decay_factor=0.5)

    training_func(dataset_name=dataset_name_training, model_dir="model_store/papyri",
                  model_name="model_papyri.pth", epochs=1000, learning_rate=5e-3,
                  initialization="model_store/artificial/model_artificial_best.pth", decay_factor=0.7, decay=125)


if __name__ == "__main__":
    main()
