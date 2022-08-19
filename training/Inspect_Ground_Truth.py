import fiftyone
from fiftyone import load_dataset

dataset = [
    "ICFHR2022_train",
    "ICFHR2022_artificial"
]

dataset = load_dataset(dataset[1])

session = fiftyone.launch_app(dataset, port=5151)

session.wait()
