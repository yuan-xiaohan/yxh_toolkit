import cv2
import numpy as np
import os
import nrrd
import gc
from utils import *

'''
load data
read data from NRRD and save to PNG in different format 
'''


# 读mask的nrrd转化为mat
def mask_nrrd2mat(mask_dir):
    readdata_mask, header_mask = nrrd.read(mask_dir)
    mask_map = np.zeros([256, 256, readdata_mask.shape[2]])
    for index in range(readdata_mask.shape[2]):
        mask_img = readdata_mask[:, :, index]
        _, mask_img = cv2.threshold(mask_img, 0, 255, cv2.THRESH_BINARY)
        mask_img = cv2.resize(mask_img, (256, 256))
        mask_img = ImgTrasform(mask_img)
        _, mask_img = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
        mask_map[:, :, index] = mask_img
    return mask_map

# 3D mat to png
def mat2png(mat, title, out_dir):
    for index in range(mat.shape[2]):
        img = mat[:, :, index]
        cv2.imwrite(out_dir + "\\" + title + "_" + changenum(index+1) + ".png", img)  # 命名为“id_time_num.png”


# 把尺寸全部改为256*256, 并输出img和mask为png格式(输出四腔室，0,1,2,3,4)
def data_load(root_nrrd, output_pic=None, output=True):
    print("Data is loading ......")
    img = np.zeros([256, 256, 1])
    mask = np.zeros([256, 256, 1])

    for root, dirs, files in os.walk(root_nrrd):
        if files != []:
            time = os.path.split(root)[1]
            id = os.path.split(os.path.split(root)[0])[1]
            print(id)
            print(time)
            in_bg_dir = os.path.join(root, files[0])
            la_a_dir = os.path.join(root, files[1])
            lv_a_dir = os.path.join(root, files[2])
            ra_a_dir = os.path.join(root, files[3])
            rv_a_dir = os.path.join(root, files[4])

            # img
            readdata_bg, header_bg = nrrd.read(in_bg_dir)
            bg_map = np.zeros([256, 256, readdata_bg.shape[2]])
            if readdata_bg.shape[0] == 512:
                for index in range(readdata_bg.shape[2]):
                    bg_img = readdata_bg[:, :, index]
                    bg_img = cv2.resize(bg_img, (256, 256))
                    bg_img = ImgTrasform(bg_img)
                    bg_img = Normalization(windowAdjust(bg_img, 800, 200)) * 255
                    bg_map[:, :, index] = bg_img
            else:
                for index in range(readdata_bg.shape[2]):
                    bg_img = readdata_bg[:, :, index]
                    bg_img = ImgTrasform(bg_img)
                    bg_img = Normalization(windowAdjust(bg_img, 800, 200)) * 255
                    bg_map[:, :, index] = bg_img
            # print(bg_map.shape)
            img = np.concatenate((img, bg_map), axis=2)


            # mask
            mask_la = mask_nrrd2mat(la_a_dir)
            mask_lv = mask_nrrd2mat(lv_a_dir)
            mask_ra = mask_nrrd2mat(ra_a_dir)
            mask_rv = mask_nrrd2mat(rv_a_dir)

            mask_la = mask_la / 255 * 1
            mask_lv = mask_lv / 255 * 2
            mask_ra = mask_ra / 255 * 3
            mask_rv = mask_rv / 255 * 4
            mask_map = mask_la + mask_lv + mask_ra + mask_rv
            # print(mask_map.shape)
            mask = np.concatenate((mask, mask_map), axis=2)

            if output == True:
                mat2png(bg_map, id + "_" + time, output_pic + "\img")
                mat2png(mask_map, id + "_" + time, output_pic + "\mask")


    img = img[:, :, 1:]  # 因为是矩阵堆叠做的，所以把第一个0矩阵去掉
    mask = mask[:, :, 1:]
    # print(img.shape, "\n", mask.shape)
    # list = range(0, img.shape[2], 2)
    # img = img[:, :, list]
    # mask = mask[:, :, list]
    print("data size:", img.shape)  # (256, 256, slice_num), float64

    return img, mask


# 把尺寸全部改为256*256, 并输出img和mask为png格式(输出一腔室，0&255)
def data_load_onechamper(root_nrrd, output_pic=None, output=True):
    print("Data is loading ......")
    # img = np.zeros([256, 256, 1])
    # mask = np.zeros([256, 256, 1])

    for root, dirs, files in os.walk(root_nrrd):
        if files != []:
            time = os.path.split(root)[1]
            id = os.path.split(os.path.split(root)[0])[1]
            print(id)
            print(time)
            in_bg_dir = os.path.join(root, files[0])
            # la_a_dir = os.path.join(root, files[1])
            lv_a_dir = os.path.join(root, files[2])
            # ra_a_dir = os.path.join(root, files[3])
            # rv_a_dir = os.path.join(root, files[4])

            # img
            readdata_bg, header_bg = nrrd.read(in_bg_dir)
            bg_map = np.zeros([256, 256, readdata_bg.shape[2]])
            if readdata_bg.shape[0] == 512:
                for index in range(readdata_bg.shape[2]):
                    bg_img = readdata_bg[:, :, index]
                    bg_img = cv2.resize(bg_img, (256, 256))
                    bg_img = ImgTrasform(bg_img)
                    bg_img = Normalization(windowAdjust(bg_img, 800, 200)) * 255
                    bg_map[:, :, index] = bg_img
            else:
                for index in range(readdata_bg.shape[2]):
                    bg_img = readdata_bg[:, :, index]
                    bg_img = ImgTrasform(bg_img)
                    bg_img = Normalization(windowAdjust(bg_img, 800, 200)) * 255
                    bg_map[:, :, index] = bg_img
            # print(bg_map.shape)
            # img = np.concatenate((img, bg_map), axis=2)


            # mask
            # mask_la = mask_nrrd2mat(la_a_dir)
            mask_lv = mask_nrrd2mat(lv_a_dir)
            # mask_ra = mask_nrrd2mat(ra_a_dir)
            # mask_rv = mask_nrrd2mat(rv_a_dir)

            # mask_la = mask_la / 255 * 1
            # mask_lv = mask_lv / 255 * 2
            # mask_ra = mask_ra / 255 * 3
            # mask_rv = mask_rv / 255 * 4
            mask_map = mask_lv
            # print(mask_map.shape)
            # mask = np.concatenate((mask, mask_map), axis=2)

            if output == True:
                mat2png(bg_map, id + "_" + time, output_pic + "\img")
                mat2png(mask_map, id + "_" + time, output_pic + "\mask\LV")

            gc.collect()


    # img = img[:, :, 1:]  # 因为是矩阵堆叠做的，所以把第一个0矩阵去掉
    # mask = mask[:, :, 1:]

    # print(img.shape, "\n", mask.shape)
    # list = range(0, img.shape[2], 2)
    # img = img[:, :, list]
    # mask = mask[:, :, list]
    # print("data size:", img.shape)  # (256, 256, slice_num), float64

    return



def main():
    # img_train, mask_train = data_load(r"F:\XiaohanYuan_Data\heart_reconstruction\train\nrrd", r"F:\XiaohanYuan_Data\heart_reconstruction\train\img")
    #     # img_val, mask_val = data_load(r"F:\XiaohanYuan_Data\heart_reconstruction\val\nrrd", r"F:\XiaohanYuan_Data\heart_reconstruction\val\img")

    # data_load_onechamper(r"Z:\XiaohanYuan\all_nrrd_new\train",
    #                                              r"F:\XiaohanYuan_Data\2D\train")
    data_load_onechamper(r"Z:\XiaohanYuan\all_nrrd_new\test",
                                             r"F:\XiaohanYuan_Data\2D\test")




    '''
    img_train, mask_train = data_load(r"F:\XiaohanYuan_Data\size255_unet\test\train")
    img_val, mask_val = data_load(r"F:\XiaohanYuan_Data\size255_unet\test\val")
    f = h5py.File(r"F:\XiaohanYuan_Data\size255_unet\size255_unet.hdf5", "w")
    g_train = f.create_group("train")
    img_train = g_train.create_dataset("img", data=img_train)
    mask_train = g_train.create_group("mask", data=mask_train)

    g_val = f.create_dataset("g_val")
    img_val = g_val.create_dataset("img", data=img_val)
    mask_val = g_val.create_group("mask", data=mask_val)

    f.close()

    f = h5py.File(r"F:\XiaohanYuan_Data\size255_unet\size255_unet.hdf5", "r")
    print(f.filename, ":")
    print([key for key in f.keys()], "\n")

    g_train = f["train"]
    print([key for key in g_train.keys()])

    img_train = f["/g_train/img"]
    print(img_train.shape)
    '''

if __name__ == "__main__":
    main()





