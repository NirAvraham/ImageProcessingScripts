import os
import shutil
import math
from tqdm import tqdm
import re
import cv2
import numpy as np

dir_path = r"C:\tmp\Idan+\Front"
output_dir = r"C:\tmp\Idan+\Front\Output"
balance_uniqe_word_file = {"Boot_Ant_USB_B":5, "Boot_Ant_USB_D":5, "Lights_A_Ant_USB_B":2, "Lights_B_USB_D":2,"Standby_USB_B":2}
mul_factor_balance = 2
saved_image_type = ".png"
regex_exp = r"\d*idan_([a-zA-Z_]+[0-9]?)_Images.\w*"


def check_relevant_words(file_name):

    p = re.compile(regex_exp)
    match = p.match(file_name)
    single_unique = match.group(1)
    for unique_word in balance_uniqe_word_file.keys():
        if unique_word == single_unique:
            return balance_uniqe_word_file[unique_word], True
    return None, False


def add_files_for_balance(dir_type, list_dir):
    for type in dir_type:
        if not os.path.exists(os.path.join(output_dir, type)):
            os.mkdir(os.path.join(output_dir, type))
    for file_name in tqdm(list_dir):
        if file_name.startswith('.'):
            continue
        balance_factor, relevant = check_relevant_words(file_name)
        if relevant:
            for i in range(balance_factor):

                img_src = os.path.join(dir_path, dir_type[0], file_name)
                img_dst = os.path.join(output_dir, dir_type[0],
                                       file_name.split(".")[0] + "_" + str(i) + saved_image_type)
                json_src = os.path.join(dir_path, dir_type[1], file_name.split(".")[0] + ".png")
                json_dst = os.path.join(output_dir, dir_type[1], file_name.split(".")[0] + "_" + str(i) + ".png")
                if os.path.exists(img_src) and os.path.exists(json_src):
                    shutil.copy2(img_src, img_dst)
                    shutil.copy2(json_src, json_dst)


def get_stats(list_dir, type):
    p = re.compile(regex_exp)
    unique_list = {}
    images_list = {}
    for file_name in tqdm(list_dir):
        if file_name.startswith('.'):
            continue
        match = p.match(file_name)
        if match:
            if match.group(1) not in unique_list:
                unique_list[match.group(1)] = 1
                images_list[match.group(1)] = cv2.imread(os.path.join(dir_path, type, file_name))
            unique_list[match.group(1)] += 1
    img_size = 150
    margin = 5
    image_num = math.ceil(math.sqrt(len(images_list)))
    tot_size = (img_size * image_num) + (image_num * margin)
    size = tot_size, tot_size, 3
    m = np.zeros(size, dtype=np.uint8)
    row = 0
    col = 0

    step = img_size + margin
    for image in images_list.keys():
        images_list[image] = cv2.resize(images_list[image], (img_size,img_size))
        cv2.putText(images_list[image], str(unique_list[image]), (int(img_size/2), int(img_size/2)),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        cv2.putText(images_list[image], image, (0, 10 + 15),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 0), 1)
        m[row : row + img_size, col : col + img_size, :] = images_list[image]
        if col + step + img_size < tot_size:
            col += step
        else:
            col = 0
            row += step
    cv2.imshow('Stats', m)
    cv2.waitKey()


def run(command):
    dir_type = ["Images", "Masks-Filtered"]
    list_dir = os.listdir(os.path.join(dir_path, dir_type[0]))
    if command is 'stats':
        get_stats(list_dir, dir_type[0])
    else:
        add_files_for_balance(dir_type, list_dir)


if __name__ == "__main__":
    # run('stats')
    run('balance')
