from fiftyone import load_dataset

dataset = load_dataset("ICFHR2022_train")

gt_field = "annotations_detections"

classes = dataset.distinct(f"{gt_field}.detections.label")
letter_statistics = dataset.count_values(f"{gt_field}.detections.label")

print("Letter Statistics")
print("Letter - No. of occurences")

for letter, count in letter_statistics.items():
    print(f"{letter}: {count}")
