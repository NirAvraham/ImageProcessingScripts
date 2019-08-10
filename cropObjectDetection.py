import os
import cv2
import json


def get_min_max_axis(x_list, y_list, height, width, scale_factor):
    # scale_factor = 2.5
    x_min = int(min(x_list))
    x_max = int(max(x_list))
    y_min = int(min(y_list))
    y_max = int(max(y_list))
    bounding_width = int((x_max - x_min) * scale_factor / 2)
    bounding_height = int((y_max - y_min) * scale_factor / 2)
    x_min -= bounding_width
    if x_min < 0:
        x_min = 0
    x_max += bounding_width
    if x_max > width:
        x_max = width
    y_min -= bounding_height
    if y_min < 0:
        y_min = 0
    y_max += bounding_height
    if y_max > height:
        y_max = height
    return x_min, y_min, x_max, y_max


json_path = r"D:\ObjectDetectionCombined\Data\Jsons"
new_json_path = r"D:\ObjectDetectionCombined\CroppedData\Jsons"
images_path = r"D:\ObjectDetectionCombined\Data\Images"
new_images_path = r"D:\ObjectDetectionCombined\CroppedData\Images"
image_type = "jpg"

for json_file in os.listdir(json_path):
    image_file_name = ".".join([json_file.split(".")[0], image_type])
    image_file_path = os.path.join(images_path, ".".join([json_file.split(".")[0], image_type]))
    if not os.path.exists(image_file_path):
        continue
    json_file_path = os.path.join(json_path, json_file)
    with open(json_file_path, "r") as json_file_data:
        data = json_file_data.read()
        try:
            json_dic = json.loads(data)
        except:
            continue
        im = cv2.imread(image_file_path)
        (height, width, depth) = im.shape

        for i, annotation in enumerate(json_dic["annotations"]):
            try:
                annotation_list = annotation['coordinates'][0] if type(annotation['coordinates'][0]) == list else \
                annotation['coordinates']
            except:
                continue

            x_list = [coordinate['x'] for coordinate in annotation_list]
            y_list = [coordinate['y'] for coordinate in annotation_list]
            if None in x_list or None in y_list or not x_list or not y_list:
                continue
            try:
                x_min, y_min, x_max, y_max = get_min_max_axis(x_list, y_list, height, width, 2.5)
                x_list = [elem - x_min for elem in x_list]
                y_list = [elem - y_min for elem in y_list]
                json_dic["annotations"][i]['coordinates'] = [[{"x": x_list[i], "y": y_list[i]} for i in range(len(x_list))]]
                with open(os.path.join(new_json_path, json_file), 'w') as outfile:
                    json.dump(json_dic, outfile)
            except:
                print(x_list, y_list, json_file)
            cv2.imwrite(os.path.join(new_images_path, image_file_name), im[y_min:y_max, x_min:x_max])


