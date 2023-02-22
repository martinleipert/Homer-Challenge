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
                  model_name="model_artificial.pth", epochs=60, decay=20, learning_rate=2e-4, decay_factor=0.5,
                  optimizer="ADAM")

    # Training using ADAM
    training_func(dataset_name=dataset_name_training, model_dir="model_store/papyri",
                  model_name="model_papyri_adam.pth", epochs=100, learning_rate=2e-4,
                  initialization="model_store/artificial/model_artificial_best.pth", decay_factor=0.5, decay=50,
                  optimizer="ADAM")

    # Finetuning
    # training_func(dataset_name=dataset_name_training, model_dir="model_store/papyri",
    #               model_name="model_papyri_adam2.pth", epochs=100, learning_rate=5e-5,
    #               initialization="model_store/papyri/model_papyri_adam_best.pth", decay_factor=0.5, decay=50,
    #               optimizer="ADAM")


if __name__ == "__main__":
    main()
