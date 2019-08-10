import os
import random
import cv2
import numpy as np, sys

def get_data():
    images_path = r"D:\Cables_With_Body\For_Training_Filtered\Train\HDMI_Black_Trapezoid_Body"
    background_path = r"D:\Cables_With_Body\Junk\Images"
    image_base_path = os.path.join(images_path, "Images")
    images_list = os.listdir(image_base_path)
    mask_base_path = os.path.join(images_path, "Masks")
    masks_list = os.listdir(mask_base_path)
    background_list = os.listdir(background_path)
    split_images = [file_name.split(".")[0] for file_name in images_list]
    split_masks = [file_name.split(".")[0] for file_name in masks_list]
    combined_images = list(set(split_images).intersection(set(split_masks)))

    for file_name in combined_images:
        image = images_list[[i for i,image_name in enumerate(images_list) if os.path.basename(image_name).startswith(file_name)][0]]
        mask = masks_list[[i for i,mask_name in enumerate(masks_list) if os.path.basename(mask_name).startswith(file_name)][0]]
        background = background_list[random.randint(0, len(background_list) - 1)]
        yield os.path.join(image_base_path,image), os.path.join(mask_base_path, mask), os.path.join(background_path, background)


def create_gaussian_pyramid(images_list, iter):
    gaussian_list = []
    for image in images_list:
        G = image.copy()
        gpA = [G.astype(float)]
        for i in range(iter):
            G = cv2.pyrDown(G.astype(float))
            gpA.append(G)
        gaussian_list.append(gpA)
    return gaussian_list


def create_laplacian_pyramid(gaussian_list, iter):
    laplacian_list = []
    for gaussian in gaussian_list:
        lpA = [gaussian[iter - 1]]
        for i in range(iter - 1, 0, -1):
            GE = cv2.pyrUp(gaussian[i])
            L = cv2.subtract(gaussian[i - 1], GE)
            lpA.append(L)
        laplacian_list.append(lpA)
    return laplacian_list


def laplacian(A, B, mask, iter = 3, kernel_size = 3):
    kernel = np.ones((kernel_size,kernel_size))
    mask = cv2.erode(mask, kernel)

    gaussian_lists = create_gaussian_pyramid([A, B], iter)
    gpA = gaussian_lists[0]
    gpB = gaussian_lists[1]

    laplacian_list = create_laplacian_pyramid([gpA, gpB], iter)
    lpA = laplacian_list[0]
    lpB = laplacian_list[1]
    LS = []
    for la, lb in zip(lpA, lpB):
        mask = cv2.resize(mask, (la.shape[0], la.shape[1]))
        LS.append(np.where(mask > 0, la, lb))
    ls_ = LS[0]
    for i in range(1, iter):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])
    return ls_



if __name__ == "__main__":
    for (image, mask, background) in get_data():
        A = cv2.imread(image)
        A = cv2.resize(A, (512, 512))
        B = cv2.imread(background)
        B = cv2.resize(B, (A.shape[0], A.shape[1]))

        mask = cv2.imread(mask)
        ret, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        mask = cv2.resize(mask, (512, 512))
        blended = laplacian(A, B, mask)
        cv2.imwrite('Output/Output' + os.path.basename(image) + '_Pyramid_blending2.jpg', blended)