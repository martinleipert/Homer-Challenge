import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import os
import re
import fiftyone
from fiftyone import Dataset as COCODataset
from fiftyone.types import COCODetectionDataset
from fiftyone.utils.data import Sample
from fiftyone import load_dataset, Detection, ImageMetadata, Detections
from fiftyone.utils.splits import random_split
import unicodedata as ud

import random
import numpy
from greek_accentuation import accentuation

import unicodedata
import re



def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESSOURCES = os.path.join(BASE_DIR, "Ressources")
BACKGROUNDS = os.path.join(RESSOURCES, "Backgrounds")
FONTS = os.path.join(RESSOURCES, "Fonts")
TEXTS = os.path.join(RESSOURCES, "Texts")

N_SAMPLES = 1000

OUTPUT_PATH = os.path.join(BASE_DIR, "Dataset")

FONT_SIZE = (70, 140)
RANDOM_FACTOR_FONT = (-5, 5)
ADD_SPACING = 10
PAGE_SIZE = (1000, 1500)
PADDING = 150

LETTERS_DOWN = [u"γ", u"η", u"μ", u"φ"]

# For transkribing the labels
d = {ord('\N{COMBINING ACUTE ACCENT}'): None}


def main():

    # region Preparation
    # Load challenge dataset to copy properties
    original_dataset = load_dataset("ICFHR2022_train")

    example_sample = original_dataset.first()

    # Create empty coco dataset
    coco_dataset = load_dataset("ICFHR2022_artificial")

    # dataset_type=COCODetectionDataset,
    # include_id=True,
    # label_field="annotations",


    # Get the names of the available fonts (they should be installed previously)
    available_fonts = []

    for dir_name in os.listdir(FONTS):
        if dir_name in [".", ".."]:
            continue

        font_dir = os.path.join(FONTS, dir_name)
        for file_name in os.listdir(font_dir):
            if not file_name.endswith(".ttf"):
                continue
            available_fonts.append(os.path.join(font_dir, file_name))

    # Get the available backgrounds and preload the images directly
    available_backgrounds = []

    for file_name in os.listdir(BACKGROUNDS):
        file_path = os.path.join(BACKGROUNDS, file_name)
        if not os.path.isfile(file_path):
            continue
        image = Image.open(file_path).convert("RGB")
        image = numpy.array(image)
        # There are few available backgrounds so preload everything
        available_backgrounds.append(image)

    # Get the available texts and load them into a big string for sampling
    all_text_string = ""

    for file_name in os.listdir(TEXTS):
        file_path = os.path.join(TEXTS, file_name)
        if not os.path.isfile(file_path):
            continue
        with open(file_path, "r", encoding="utf8") as openfile:
            file_text = openfile.read()
            all_text_string += file_text

    # Cleanup text
    all_text_string = all_text_string.replace("\n", " ")

    all_text_string = strip_accents(all_text_string)
    for letter in [":", ";", '᾽', '‘']:
        all_text_string = all_text_string.replace(letter, "")

    # Remove duplicate whitespaces
    all_text_string = " ".join(re.split("\s+", all_text_string, flags=re.UNICODE))
    # all_text_string.replace("‘", '').replace("’", '').replace("'", '')

    textlen = len(all_text_string)

    # endregion Preparation

    # region Creation Loop
    # Here the creation process begins

    for artificial_index in range(N_SAMPLES):
        page_size_x = random.randint(PAGE_SIZE[0], PAGE_SIZE[1])
        page_size_y = random.randint(PAGE_SIZE[0], PAGE_SIZE[1])

        font_size = random.randint(FONT_SIZE[0], FONT_SIZE[1])

        # Background should be larger than image otherwise we would have a problem with this method
        background_index = random.randint(0, len(available_backgrounds)-1)
        selected_background = available_backgrounds[background_index]

        background_x_start = random.randint(0, selected_background.shape[1]-page_size_x)
        background_y_start = random.randint(0, selected_background.shape[0]-page_size_y)

        background_patch = selected_background[
                           background_y_start:(page_size_y+background_y_start),
                           background_x_start:(page_size_x+background_x_start)]

        # Calculate the grid
        x_start = PADDING # + font_size / 2
        x_end = page_size_x - PADDING # font_size / 2
        x_width = x_end - x_start
        n_letters_x = int(x_width / (font_size + ADD_SPACING))
        x_range = numpy.linspace(x_start, x_end - PADDING, n_letters_x).astype(numpy.int64)

        y_start = PADDING # - font_size / 2
        y_end = page_size_y - PADDING - font_size # / 2
        y_width = y_end - y_start
        n_letters_y = int(y_width / (font_size + ADD_SPACING))
        y_range = numpy.linspace(y_start, y_end, n_letters_y).astype(numpy.int64)

        n_letters = n_letters_x * n_letters_y

        X_positions, Y_positions = numpy.meshgrid(x_range, y_range)
        X_positions = list(X_positions.flatten())
        Y_positions = list(Y_positions.flatten())

        letter_start = random.randint(0, textlen - n_letters - 1)
        letters = all_text_string[letter_start:(letter_start + n_letters)]

        positions = zip(X_positions, Y_positions, letters)

        """
        fig = pyplot.figure(figsize=(x_width/100, y_width/100), dpi=100)
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.imshow(background_patch)
        """
        img = Image.fromarray(background_patch)
        draw = ImageDraw.Draw(img)

        # Every document in another font to simulate different writers
        font_index = random.randint(0, len(available_fonts) - 1)
        selected_font = available_fonts[font_index]

        font = ImageFont.truetype(selected_font)

        detections = []

        for index, (x_pos, y_pos, letter) in enumerate(positions):
            detection = Detection()

            x_add = random.randint(RANDOM_FACTOR_FONT[0], RANDOM_FACTOR_FONT[1])
            x_pos = x_add + x_pos
            y_add = random.randint(RANDOM_FACTOR_FONT[0], RANDOM_FACTOR_FONT[1])
            y_pos = y_add + y_pos
            letter_font_size = font_size + random.randint(RANDOM_FACTOR_FONT[0], RANDOM_FACTOR_FONT[1])

            text_color = (random.randint(0, 63), random.randint(0, 63), random.randint(0, 63))

            draw.text((x_pos, y_pos), letter, text_color, font=font.font_variant(size=letter_font_size))


            """
            annotations[index] = {
                "bbox": (x_pos+letter_font_size, y_pos+letter_font_size, letter_font_size, letter_font_size),
                "label": letter
            }
    
            ax.text(x_pos, y_pos, letter, {
                "fontfamily": selected_font,
                "fontsize": letter_font_size,
                "fontweight": random.randint(500, 800),
    
            })
            """

            if letter in [" ", ","]:
                continue

            bbox_padding = 0.15 * font_size
            x_pos = x_pos - bbox_padding
            y_pos = y_pos
            bbox_width = font_size + bbox_padding
            bbox_height = font_size + bbox_padding

            if letter in LETTERS_DOWN:
                y_add = font_size * 0.333
                # y_pos += y_add
                bbox_height += y_add

            # Calculate positions in bounding box
            # y_pos = y_pos - font_size
            x_pos = (x_pos) / page_size_x
            y_pos = (y_pos) / page_size_y

            bbox_width = bbox_width / page_size_x
            bbox_height = bbox_height / page_size_y

            bbox = [x_pos, y_pos, bbox_width, bbox_height]

            # Remove minuskels and diacrits
            # letter = ud.normalize('NFD', letter).upper().translate(letter)

            detection.label = letter.upper()
            detection.bounding_box = bbox

            detections.append(detection)
            pass

        pass

        # Create output path
        file_index = artificial_index
        output_filename = f"img_i{file_index}.jpg"
        output_dirname = f"img_i{file_index}"

        output_dir = os.path.join(OUTPUT_PATH, "images", output_dirname)
        output_file = os.path.join(output_dir, output_filename)

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        img.save(output_file)

        img_metadata = ImageMetadata()
        img_metadata.height = int(page_size_y)
        img_metadata.width = int(page_size_x)
        img_metadata.num_channels = None
        img_metadata.size_bytes = None
        img_metadata.mime_type = None

        annotations_detections = Detections()
        annotations_detections.detections.extend(detections)

        img_sample = Sample(output_file, metadata=img_metadata)
        # img_sample.set_field("metadata", )
        img_sample.set_field("media_type", "image")
        img_sample.set_field("annotations_detections", annotations_detections)
        # img_sample.compute_metadata(True)
        # img_sample.set_field("tags", ("font_name", os.path.basename(selected_font).rstrip(".ttf")))
        coco_dataset.add_sample(img_sample)

    random_split(coco_dataset, {"eval": 0.1, "train": 0.9})
    coco_dataset.save()

    # endregion Creation Loop


if __name__ == "__main__":
    main()
