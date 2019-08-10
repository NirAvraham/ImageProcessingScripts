import os
import cv2
import json
import numpy as np
from tqdm import tqdm


cables = ["HDMI", "LAN", "Power"]
for cable in tqdm(cables):
    path = os.path.join(r"C:\tmp\Create_masks\New_Masks", cable, "Jsons")
    output = os.path.join(r"C:\tmp\Create_masks\New_Masks", cable, "Debug_Masks")
    os.makedirs(output, exist_ok=True)
    # output = r"C:\tmp\Create_masks\Power\Masks"
    for j in os.listdir(path):
        with open(os.path.join(path, j)) as json_file:
            data = json.load(json_file)
            height = 1080
            width = 1920
            mask = np.zeros((width, height))
            try:
                for ann in data['annotations']:

                    if ann['label'] == "Cable_Head":
                        min_x = int(ann['coordinates'][0]['x'])
                        min_y = int(ann['coordinates'][0]['y'])
                        max_x = int(ann['coordinates'][1]['x'])
                        max_y = int(ann['coordinates'][1]['y'])

                    if ann['label'] == "Cable_Body":
                        points = []
                        for point in ann['coordinates'][0]:
                            points.append([int(point['x']), int(point['y'])])
                        # pts = np.array(points, np.int32)
                        # pts = pts.reshape((-1, 1, 2))


                pts = np.array(points)
                cv2.fillPoly(mask, pts=[pts], color=(1, 1, 1))
                # cv2.circle(mask, (min_x, min_y), 5, (2, 2, 2), -1)
                # cv2.circle(mask, (min_x, max_y), 5, (2, 2, 2), -1)
                # cv2.circle(mask, (max_x, min_y), 5, (2, 2, 2), -1)
                # cv2.circle(mask, (max_x, max_y), 5, (2, 2, 2), -1)
                cv2.rectangle(mask, (min_x, min_y), (max_x, max_y), (2, 2, 2), 5)
            except:
                print(j + " Failed")
                continue


            # cv2.imshow("mask", cv2.resize(mask * 50, dsize=None, fx=0.5, fy=0.5))
            # cv2.waitKey(0)
            name = j.split('.')[0] + '.jpg'
            cv2.imwrite(os.path.join(output, name), mask)