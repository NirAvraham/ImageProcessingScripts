import cv2
import os
import random
import numpy as np

images_path = r"C:\tmp\Off_Device_HDMI_FIltered\Output\Images_Edit"
masks_path = r"C:\tmp\Off_Device_HDMI_FIltered\Output\Net1\hdmi\Masks_Jpg"
save_dir = r"C:\tmp\Off_Device_HDMI_FIltered\Output\Check_Random_Crop_Data\Data"
rect_list = []


def check_and_create_dirs(dirs_dict):
    dirs = os.listdir(save_dir)
    if "Images" not in dirs:
        os.mkdir(dirs_dict["Images"])
    if "Masks" not in dirs:
        os.mkdir(dirs_dict["Masks"])
    return dirs_dict


def main():
    dirs_dict = {}
    dirs_dict["Images"] = os.path.join(save_dir, "Images")
    dirs_dict["Masks"] = os.path.join(save_dir, "Masks")
    Images = []
    Masks = []
    # check_and_create_dirs(dirs_dict)
    for image_name in os.listdir(images_path):
        # print(image_name)
        image = cv2.imread(os.path.join(images_path, image_name))
        mask = cv2.imread(os.path.join(masks_path, image_name), cv2.IMREAD_GRAYSCALE)
        Images.append(image)
        Masks.append(mask)
        # cv2.imwrite(os.path.join(dirs_dict["Masks"], image_name), mask)
        # cv2.imwrite(os.path.join(dirs_dict["Images"], image_name), image)

        rows, cols = mask.shape
        _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

        _, cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # bounding_rect_center, shape, _ = cv2.minAreaRect(cnts[0])
        x, y, w, h = cv2.boundingRect(cnts[0])
        cropped_image_num = 2
        padding = 300
        for i in range(cropped_image_num):
            try:
                if i < int(cropped_image_num * 0.5):
                    left_new_x = random.randint(0, x)
                    right_new_x = random.randint(x + w, cols - 1)
                    top_new_y = random.randint(0, y)
                    bottom_new_y = random.randint(y + h, rows - 1)
                else:
                    left_new_x = random.randint(0, cols - (padding + 1))
                    right_new_x = random.randint(left_new_x + padding + 1, cols - 1)
                    top_new_y = random.randint(0, rows - (padding + 1))
                    bottom_new_y = random.randint(top_new_y + padding + 1, rows - 1)
            except:
                continue

            Images.append(image[top_new_y:bottom_new_y, left_new_x:right_new_x])
            Masks.append(np.where(mask < 255, mask, 1)[top_new_y:bottom_new_y, left_new_x:right_new_x])
            # cv2.imwrite(os.path.join(dirs_dict["Masks"], image_name.split(".")[0] + "_" + str(i) + ".png"),
            #             np.where(mask < 255, mask, 1)[top_new_y:bottom_new_y, left_new_x:right_new_x])
            # cv2.imwrite(os.path.join(dirs_dict["Images"], image_name.split(".")[0] + "_" + str(i) + ".png"),
            #             image[top_new_y:bottom_new_y, left_new_x:right_new_x])
            # cv2.imshow("mask", cv2.resize(mask[top_new_y:bottom_new_y, left_new_x:right_new_x], dsize=None, fx=0.5, fy=0.5))
            # cv2.imshow("image", cv2.resize(image[top_new_y:bottom_new_y, left_new_x:right_new_x], dsize=None, fx=0.5, fy=0.5))
            # cv2.waitKey(0)
    num = len(Images)
    train_path = os.path.join(save_dir, "Train")
    os.mkdir(os.path.join(train_path, "Images"))
    os.mkdir(os.path.join(train_path, "Masks"))
    valid_path = os.path.join(save_dir, "Validation")
    os.mkdir(os.path.join(valid_path, "Images"))
    os.mkdir(os.path.join(valid_path, "Masks"))
    test_path = os.path.join(save_dir, "Test")
    os.mkdir(os.path.join(test_path, "Images"))
    os.mkdir(os.path.join(test_path, "Masks"))
    for i in range(len(Images)):
        if i < int(0.6 * num):
            cv2.imwrite(os.path.join(os.path.join(train_path, "Images"), str(i) + ".png"), Images[i])
            cv2.imwrite(os.path.join(os.path.join(train_path, "Masks"), str(i) + ".png"), Masks[i])
        elif i < int(0.8 * num):
            cv2.imwrite(os.path.join(os.path.join(valid_path, "Images"), str(i) + ".png"), Images[i])
            cv2.imwrite(os.path.join(os.path.join(valid_path, "Masks"), str(i) + ".png"), Masks[i])
            print(num, " ",i)
        else:
            cv2.imwrite(os.path.join(os.path.join(test_path, "Images"), str(i) + ".png"), Images[i])
            cv2.imwrite(os.path.join(os.path.join(test_path, "Masks"), str(i) + ".png"), Masks[i])

    print("")


if __name__ == "__main__":
    main()
