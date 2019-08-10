import os
import cv2
import json
from enum import Enum
import numpy as np


class Attributes(Enum):
    OFF = 1
    GREEN = 2
    BLUE = 3
    RED = 4
    UNKNOWN = 5


def get_center(mask):
    moments = cv2.moments(mask)
    if not moments["m00"]:
        return None
    cx = int(moments["m10"] / (moments["m00"] + 1e-5))
    cy = int(moments["m01"] / (moments["m00"] + 1e-5))
    return cx, cy


def get_center_label_tuple(mask):
    centers = []
    labels = []
    _, cnt_data, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(cnt_data, key=lambda x: cv2.contourArea(x), reverse=True)

    for i, cnt in enumerate(contours):
        cnt_mask = np.zeros((mask.shape[0], mask.shape[1]))
        cv2.drawContours(cnt_mask, contours, i, (255, 255, 255), -1)
        _, cnt_mask = cv2.threshold(cnt_mask, 127, 255, cv2.THRESH_BINARY)
        cnt_mask = cnt_mask.astype("uint8")
        cx, cy = get_center(cnt_mask)
        centers.append((cx, cy))
        label = Attributes(mask[cy, cx])
        labels.append(label)
    return centers, labels


def create_json(mask, file_name):
    centers, labels = get_center_label_tuple(mask)
    coordinates = {}
    all_dict = {"_id": "5c2860f9d3b804001ab0d1de", 'filename': file_name, 'annotations': []}
    for i, (x, y) in enumerate(centers):
        coordinates['x'] = float(x)
        coordinates['y'] = float(y)
        coordinates['z'] = float(0)
        attributes = [] if labels[i] == Attributes.OFF else [labels[i].name.lower()]
        metadata = {"system":{"status": "issue" if labels[i] == Attributes.UNKNOWN else None}}
        dict_tags = {"label": "port", "type": "point", 'coordinates': coordinates.copy(),
                     "attributes": attributes, "metadata":metadata}
        all_dict['annotations'].append(dict_tags)
    return all_dict


def run_dir_images_masks(dir_path):
    jsons_path = os.path.join(dir_path, "Jsons")
    os.makedirs(jsons_path, exist_ok=True)
    masks_path = os.path.join(dir_path, "Masks")
    for mask_name in os.listdir(masks_path):
        mask_full_path = os.path.join(masks_path, mask_name)
        mask = cv2.imread(mask_full_path, cv2.IMREAD_GRAYSCALE)
        json_dict = create_json(mask, mask_name)
        json_filename = os.path.join(jsons_path, mask_name.split('.')[0] + '.json')
        with open(json_filename, 'w') as json_file:
            json.dump(json_dict, json_file)


if __name__ == "__main__":
    path_masks = r"C:\Users\nira\Desktop\check"
    run_dir_images_masks(path_masks)

