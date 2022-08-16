import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot
from PIL import Image
import os
import re

import random
import numpy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESSOURCES = os.path.join(BASE_DIR, "Ressources")
BACKGROUNDS = os.path.join(RESSOURCES, "Backgrounds")
FONTS = os.path.join(RESSOURCES, "Fonts")
TEXTS = os.path.join(RESSOURCES, "Texts")

N_SAMPLES = 1
# 1000

OUTPUT_PATH = os.path.join(BASE_DIR, "Dataset")

FONT_SIZE = (70, 140)
RANDOM_FACTOR_FONT = (-5, 5)
ADD_SPACING = 10
PAGE_SIZE = (1000, 1500)
PADDING = 150


def main():

    # Get the names of the available fonts (they should be installed previously)
    available_fonts = []

    for dir_name in os.listdir(FONTS):
        if dir_name in [".", ".."]:
            continue
        available_fonts.append(dir_name)

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
    textlen = len(all_text_string)

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
    x_start = PADDING
    x_end = page_size_x - PADDING
    x_width = x_end - x_start
    n_letters_x = int(x_width / (font_size + ADD_SPACING))
    x_range = numpy.linspace(x_start, x_end - PADDING, n_letters_x).astype(numpy.int64)

    y_start = PADDING
    y_end = page_size_y - PADDING
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

    fig = pyplot.figure(figsize=(x_width/100, y_width/100), dpi=100)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.imshow(background_patch)

    annotations = {}

    for index, (x_pos, y_pos, letter) in enumerate(positions):

        x_add = random.randint(RANDOM_FACTOR_FONT[0], RANDOM_FACTOR_FONT[1])
        x_pos = x_add + x_pos
        y_add = random.randint(RANDOM_FACTOR_FONT[0], RANDOM_FACTOR_FONT[1])
        y_pos = y_add + y_pos
        letter_font_size = font_size + random.randint(RANDOM_FACTOR_FONT[0], RANDOM_FACTOR_FONT[1])

        # Every letter in another font
        font_index = random.randint(0, len(available_fonts) - 1)
        selected_font = available_fonts[font_index]

        annotations[index] = {
            "bbox": (x_pos+letter_font_size, y_pos+letter_font_size, letter_font_size, letter_font_size),
            "label": letter
        }

        ax.text(x_pos, y_pos, letter, {
            "fontfamily": selected_font,
            "fontsize": letter_font_size,
            "fontweight": random.randint(500, 800),

        })


        pass

    pass


if __name__ == "__main__":
    main()
