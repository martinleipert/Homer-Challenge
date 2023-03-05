import json
from pathlib import Path
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("filepath", help="Path to file to convert from binary to text")

args = parser.parse_args()

path = Path(args.filepath)

with open(path, "rb") as openfile:
    raw_text = openfile.read()

raw_text = raw_text.decode("utf-8")
json_dict = json.loads(raw_text)


# Remove lists from tags

new_annotations = []

for item in json_dict["annotations"]:

    new_item = {}

    for ikey, ivalue in item.items():

        # ignore the tags => causes errors
        if ikey == "tags":
            continue

        new_item[ikey] = ivalue

    new_annotations.append(new_item)


json_dict["annotations"] = new_annotations

path_2 = os.path.join(path.parent, f"{path.stem}_Copy.json")

with open(path_2, "w") as outfile:
    outfile.write(json.dumps(json_dict, indent=4))
