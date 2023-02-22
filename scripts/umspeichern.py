import json
from pathlib import Path
import os
import sys

PATH = r"C:\Users\ltm\PycharmProjects\HomerChallenge\dataset\HomerCompTesting\HomerCompTestingReadCoco.json"
PATH = Path(PATH)

with open(PATH, "rb") as openfile:
    raw_text = openfile.read()

raw_text = raw_text.decode("utf-8")
json_dict = json.loads(raw_text)

path_2 = os.path.join(PATH.parent, f"{PATH.stem}_Copy.json")

with open(path_2, "w") as outfile:
    json.dump(json_dict, outfile)
