import fiftyone
from fiftyone import load_dataset

dataset = load_dataset("ICFHR2022_train")

session = fiftyone.launch_app(dataset, port=5151)

session.wait()
