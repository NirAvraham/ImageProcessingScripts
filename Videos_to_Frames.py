import os

import cv2
import random
import numpy as np

path = r"D:\Assurance Project\Philips\UnbiasedData\Videos"
output_path = os.path.join(path, "Images")
os.makedirs(output_path, exist_ok=True)

files_list = os.listdir(path)

frame_rate = 5

frames_to_save = 550
all_frames = []
for file_name in files_list:
    full_path = os.path.join(path, file_name)
    print(file_name)
    if os.path.isdir(full_path):
        continue
    video = cv2.VideoCapture(full_path)

    frame_cnt = 0
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            if frame_cnt % frame_rate == 0:
                frame = np.rot90(frame, k=3)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                all_frames.append(frame)
        else:
            break
        frame_cnt += 1
    video.release()


random.shuffle(all_frames)

for i, image in enumerate(all_frames[:frames_to_save]):
    image_path = os.path.join(output_path, str(i) + ".png")
    cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


