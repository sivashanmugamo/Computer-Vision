"""
Character Detection

The goal is to find all of the coordinates where a specific character appears using template matching.

There are 3 sub tasks:
1. Detect character 'a'.
2. Detect character 'b'.
3. Detect character 'c'.
"""

import argparse
import json
import os

import utils
from task1 import *

def parse_args():
    parser = argparse.ArgumentParser(description="edge-character detection")
    parser.add_argument( "--img_path", 
        type=str, 
        default="./data/characters.jpg",
        help="path to the image used for character detection")
    parser.add_argument( "--template_path", 
        type=str, 
        default="",
        choices=["./data/a.jpg", "./data/b.jpg", "./data/c.jpg"],
        help="path to the template image")
    parser.add_argument( "--result_saving_directory",
        dest="rs_directory",
        type=str,
        default="./results/",
        help="directory to which results are saved")
    args = parser.parse_args()
    return args

def detect(img, template):
    """
    Detect a given character, i.e., the character in the template image.

    Args:
        img: nested list (int), image that contains character to be detected.
        template: nested list (int), template image.

    Returns:
        coordinates: list (tuple), a list whose elements are coordinates where the character appears.
            format of the tuple: (x (int), y (int)), x and y are integers.
            x: row that the character appears (starts from 0).
            y: column that the character appears (starts from 0).
    """

    template_h = len(template)
    template_w = max([len(i) for i in template])

    image_h = len(img)
    image_w = max([len(i) for i in img])

    output = []
    threshold = 0.85

    for i in range(image_h - template_h):
        temp = []
        for j in range(image_w - template_w):
            cropped_img = img[i: i + template_h]
            cropped_img = [x[j: j+template_w] for x in cropped_img]

            i_0_bar = sum([sum(x) for x in template])/(template_h * template_w)

            i_1_bar = sum([sum(y) for y in cropped_img])/(template_h * template_w)

            template_sigma = (sum([sum(v) for v in [[(w-i_0_bar)**2 for w in x] for x in template]])) ** 0.5

            image_sigma = (sum([sum(v) for v in [[(w-i_1_bar)**2 for w in x] for x in cropped_img]])) ** 0.5

            '''Normalized cross correlation calculation'''
            e_ncc = sum([(template[x][w] - i_0_bar) * (cropped_img[x][w] - i_1_bar) for w in range(template_w) for x in
                         range(template_h)]) / (template_sigma * image_sigma)

            if str(e_ncc) == 'nan':
                temp.append(0)
            else:
                temp.append(e_ncc)
        output.append(temp)

    coordinates = []
    for x in range(len(output)):
        for y in range(len(output[0])):
            coordinates_temp = []
            if output[x][y] >= threshold:
                coordinates_temp.append(x)
                coordinates_temp.append(y)
                coordinates.append(coordinates_temp)
    
    return coordinates

def save_results(coordinates, template, template_name, rs_directory):
    results = {}
    results["coordinates"] = sorted(coordinates, key=lambda x: x[0])
    results["template_size"] = (len(template), len(template[0]))
    with open(os.path.join(rs_directory, template_name), "w") as file:
        json.dump(results, file)

def main():
    args = parse_args()

    img = read_image(args.img_path)
    template = read_image(args.template_path)

    coordinates = detect(img, template)

    template_name = "{}.json".format(os.path.splitext(os.path.split(args.template_path)[1])[0])
    save_results(coordinates, template, template_name, args.rs_directory)

if __name__ == "__main__":
    main()
