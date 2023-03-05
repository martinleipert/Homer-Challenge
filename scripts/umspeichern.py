import json
from pathlib import Path
import os

PATH = r"C:\Users\ltm\PycharmProjects\Homer-Challenge\dataset\HomerCompTraining\HomerCompTrainingReadCoco.json"
PATH = Path(PATH)

with open(PATH, "rb") as openfile:
    raw_text = openfile.read()

raw_text = raw_text.decode("utf-8")
json_dict = json.loads(raw_text)

path_2 = os.path.join(PATH.parent, f"{PATH.stem}_Copy.json")

with open(path_2, "w") as outfile:
    outfile.write(json.dumps(json_dict, indent=4))
