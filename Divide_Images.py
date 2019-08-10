import os
import shutil
import random

data_path = r"D:\Assurance Project\Device_Classifier"
output_path = r"C:\ML_Training\Classification\Assurance_Device_Classifier\Data"

dirs_list = ["Train", "Validation", "Test"]


for d_path in dirs_list:
    new_path = os.path.join(output_path, d_path)
    os.makedirs(new_path, exist_ok=True)
    for c in os.listdir(data_path):
        os.makedirs(os.path.join(new_path,c), exist_ok=True)

for c in os.listdir(data_path):
    images_list = os.listdir(os.path.join(data_path, c))
    random.shuffle(images_list)

    for i, name in enumerate(images_list):

        if i < int(len(images_list) * 0.6):
            shutil.copy2(os.path.join(data_path, c, name), os.path.join(output_path, dirs_list[0], c, name))

        elif i < int(len(images_list) * 0.8):
            shutil.copy2(os.path.join(data_path, c, name), os.path.join(output_path, dirs_list[1], c, name))

        else:
            shutil.copy2(os.path.join(data_path, c, name), os.path.join(output_path, dirs_list[2], c, name))

