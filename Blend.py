import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_pyr_up(shape, pyrUp_A):
    height, width = shape[:2]
    return cv2.resize(pyrUp_A, (width, height))


def Laplacian_Pyramid_Blending_with_mask(A, B, m, num_levels = 6):
    # assume mask is float32 [0,1]

    # generate Gaussian pyramid for A,B and mask
    GA = A.copy()
    GB = B.copy()
    GM = m.copy()
    gpA = [GA]
    gpB = [GB]
    gpM = [GM]
    for i in range(num_levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(np.float32(GA))
        gpB.append(np.float32(GB))
        gpM.append(np.float32(GM))
    # generate Laplacian Pyramids for A,B and masks
    lpA  = [gpA[num_levels-1]] # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lpB  = [gpB[num_levels-1]]
    gpMr = [gpM[num_levels-1]]
    for i in range(num_levels-1, 0, -1):
        # Laplacian: subtarct upscaled version of lower level from current level
        # to get the high frequencies
        pyrUp_A = cv2.pyrUp(gpA[i])
        pyrUp_B = cv2.pyrUp(gpB[i])
        LA = np.subtract(gpA[i-1], get_pyr_up(gpA[i-1].shape, pyrUp_A))
        LB = np.subtract(gpB[i-1], get_pyr_up(gpB[i-1].shape, pyrUp_B))
        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i-1]) # also reverse the masks
    # Now blend images according to mask in each level
    LS = []
    for la,lb,gm in zip(lpA,lpB,gpMr):
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)
    # now reconstruct
    ls_ = LS[0]
    for i in range(1,num_levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(get_pyr_up(LS[i].shape, ls_), LS[i])

    return ls_

def get_data():
    images_path = r"C:\Users\nira\PycharmProjects\Junk\For_Training\Train\HDMI_Black_Generic"
    background_path = r"C:\Users\nira\Downloads\Images"
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
        background = background_list[random.randint(0, len(background_list))]
        yield os.path.join(image_base_path,image), os.path.join(mask_base_path, mask), os.path.join(background_path, background)







if __name__ == '__main__':
    size = 514
    for (image, mask, background) in get_data():
        A = cv2.imread(image)
        A = cv2.cvtColor(A, cv2.COLOR_BGR2HSV)
        A = cv2.resize(A, (size, size))
        B = cv2.imread(background)
        B = cv2.cvtColor(B, cv2.COLOR_BGR2HSV)
        B = cv2.resize(B, (size, size))
        m = cv2.imread(mask)
        # m = cv2.cvtColor(m, cv2.COLOR_BGR2HSV)
        m = cv2.resize(m, (size, size))
        ret, m = cv2.threshold(m,0,255, cv2.THRESH_BINARY)


        # axarr[1, 1].imshow(image_datas[3])


        # fig = plt.figure(figsize=(8,8))
        # fig.add_subplot(1, 1, 1)
        # plt.imshow(cv2.resize(A, (100,100)))
        # fig.add_subplot(1, 2, 2)
        # plt.imshow(cv2.resize(B, (100,100)))
        # fig.add_subplot(1, 3, 3)
        # plt.imshow(cv2.resize(m, (100,100)))

        # cv2.imshow("mask", m)
        # cv2.waitKey(0)
        kernel = np.ones((3,3), np.uint8)
        m = cv2.erode(m, kernel, iterations=1)
        # ret, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
        # m = np.zeros_like(A, dtype='float32')
        m = cv2.normalize(m, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # m[:,A.shape[1]/2:] = 1 # make the mask half-and-half
        lpb = Laplacian_Pyramid_Blending_with_mask(A[:,:,0], B[:,:,0], m[:,:,0], 4)

        # cv2.imshow("lpb", A)
        # cv2.waitKey(0)
        # cv2.imwrite("Images/HDMI_Output_2.png",lpb)
        f, axarr = plt.subplots(2, 2)
        size = 400
        axarr[0, 0].imshow(cv2.cvtColor(cv2.resize(A, (size, size)), cv2.COLOR_HSV2RGB))
        axarr[0, 1].imshow(cv2.cvtColor(cv2.resize(B, (size, size)), cv2.COLOR_HSV2RGB))
        axarr[1, 0].imshow(cv2.cvtColor(cv2.resize(m, (size, size)), cv2.COLOR_HSV2RGB))
        # A[:, :, 0] = lpb
        axarr[1, 1].imshow(np.uint8(lpb))
        plt.show()
        # cv2.imshow("Image", lpb)
        # cv2.waitKey(0)