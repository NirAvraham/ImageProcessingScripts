import os
import cv2
import numpy as np
import random

images_path = r"C:\tmp\Idan+\Junk\Test"
output_path = r"C:\tmp\Idan+\Junk\Test_Output"


def main():

    for image_name in os.listdir(images_path):
        file_path = os.path.join(images_path, image_name)
        image = cv2.imread(file_path)
        rows, cols, _ = image.shape
        black_image = np.zeros((rows, cols))
        white_rect_size = 70
        # pos_x = random.randint(white_rect_size + 1, cols - white_rect_size - 1)
        # pos_y = random.randint(white_rect_size + 1, rows - white_rect_size - 1)
        black_image[white_rect_size:-white_rect_size,white_rect_size:-white_rect_size] = 1
        # cv2.imshow("Asdf", black_image * 255)
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join(output_path, image_name), black_image)


if __name__ == '__main__':
    main()