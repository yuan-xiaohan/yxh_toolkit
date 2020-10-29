import cv2
from PIL import Image
import nrrd
import numpy as np
import os
import glob
from nrrd2png_3plane import *

# 反变换：水平镜像 + 逆时针90度旋转
def Inverse_ImgTrasform(img):
    row, col = img.shape[:2]
    img = cv2.flip(img, 1)
    M = cv2.getRotationMatrix2D((col / 2, row / 2), 90, 1)
    img_new =cv2.warpAffine(img, M, (col, row))
    return img_new


root_nrrd = "E:\WHS_10_nrrd"
# root_png = "E:\\WHS_10_png"
root_png = "E:\WHS_10_png"

for pair in [["chnxiaoqing","01"],["huxiaoying","02"],["jiangzhongyin","03"],["lanjunfang","04"],["liyanping","05"],["shadebin","06"],["wuyong","07"],["xialianglluan","08"],["zhuyuyin","09"]]:
    pname = pair[0]
    pnumber = pair[1]
    print(pname,pnumber)

    # pname = "huxiaoying" #病人名字
    # pnumber = "02" #病人序号

    for time in ["010","020","030","040","050","060","070","080","090","100"]:
        print(time)
        bg_name = "BG.nrrd" #背景nrrd名字 !

        # 背景
        in_bg_dir = os.path.join(root_nrrd, pname, time, bg_name)  # 背景nrrd路径
        os.mkdir(os.path.join(root_png, pname))  # 新建文件夹,一个人建好后这句话要注释掉 ！
        os.mkdir(os.path.join(root_png, pname, time))  # 新建文件夹

        os.mkdir(os.path.join(root_png, pname, time, "BG"))  # 新建背景文件夹
        os.mkdir(os.path.join(root_png, pname, time, "BG", "A"))  # 新建A轴文件夹
        os.mkdir(os.path.join(root_png, pname, time, "BG", "S"))  # 新建S轴文件夹
        os.mkdir(os.path.join(root_png, pname, time, "BG", "C"))  # 新建C轴文件夹
        out_bg_a_dir = os.path.join(root_png, pname, time, "BG", "A")
        out_bg_s_dir = os.path.join(root_png, pname, time, "BG", "S")
        out_bg_c_dir = os.path.join(root_png, pname, time, "BG", "C")

        readdata_bg, header_bg = nrrd.read(in_bg_dir)

        # nrrd2png
        for index in range(readdata_bg.shape[2]):
            map_bg = readdata_bg[:, :, index]
            map_bg = ImgTrasform(map_bg)
            map_bg = Normalization(windowAdjust(map_bg, 800, 200)) * 255
            cv2.imwrite(out_bg_a_dir + "\\" + pnumber + "_" + time + "_" + "BG" + "_A_" + changenum(index + 1) + ".png", map_bg)  # 命名为“pname_time_BG_A_+序号.png”

        # axial to sagittal and coronal
        # sorted_bg = ContentSort(out_bg_a_dir)
        bg_a = png2mat(out_bg_a_dir, os.listdir(out_bg_a_dir))
        bg_s = AxialToSagittal(bg_a)
        mat2png(bg_s, "\\" + pnumber + "_" + time + "_" + "BG" + "_S_", out_bg_s_dir)
        bg_c = AxialToCoronal(bg_a)
        mat2png(bg_c, "\\" + pnumber + "_" + time + "_" + "BG" + "_C_", out_bg_c_dir)

        # mask
        for champer in ["LA","LV","RA","RV"]:
            in_mask_dir = os.path.join(root_nrrd, pname, time) + "\\" + champer + "-label.nrrd"  # 掩膜nrrd路径
            os.mkdir(os.path.join(root_png, pname, time, champer))  # 新建掩膜文件夹
            os.mkdir(os.path.join(root_png, pname, time, champer, "A")) #新建A轴文件夹
            os.mkdir(os.path.join(root_png, pname, time, champer, "S"))  # 新建S轴文件夹
            os.mkdir(os.path.join(root_png, pname, time, champer, "C"))  # 新建C轴文件夹
            out_mask_a_dir = os.path.join(root_png, pname, time, champer, "A")
            out_mask_s_dir = os.path.join(root_png, pname, time, champer, "S")
            out_mask_c_dir = os.path.join(root_png, pname, time, champer, "C")


            readdata_mask, header_mask = nrrd.read(in_mask_dir)

            # nrrd2png
            for index in range(readdata_mask.shape[2]):
                map_mask = readdata_mask[:, :, index]
                ret, map_mask = cv2.threshold(map_mask, 0, 255, cv2.THRESH_BINARY)  # 二值化
                map_mask = ImgTrasform(map_mask)
                cv2.imwrite(out_mask_a_dir + "\\" + pnumber + "_" + time + "_" + champer + "_A_" + changenum(index + 1) + ".png", map_mask)  # 命名为“LV_A_+序号.png”

            # axial to sagittal and coronal
            # sorted_mask = ContentSort(out_mask_a_dir)
            mask_a = png2mat(out_mask_a_dir, os.listdir(out_mask_a_dir))
            mask_s = AxialToSagittal(mask_a)
            mat2png(mask_s, "\\" + pnumber + "_" + time + "_" + champer + "_S_", out_mask_s_dir)
            mask_c = AxialToCoronal(mask_a)
            mat2png(mask_c, "\\" + pnumber + "_" + time + "_" + champer + "_C_", out_mask_c_dir)


        # overlap
        la_a_dir = os.path.join(root_png, pname, time, "LA", "A")
        lv_a_dir = os.path.join(root_png, pname, time, "LV", "A")
        ra_a_dir = os.path.join(root_png, pname, time, "RA", "A")
        rv_a_dir = os.path.join(root_png, pname, time, "RV", "A")
        out_la_a_dir = os.path.join(root_png, pname, time, "LA", "A")
        out_la_s_dir = os.path.join(root_png, pname, time, "LA", "S")
        out_la_c_dir = os.path.join(root_png, pname, time, "LA", "C")
        out_rv_a_dir = os.path.join(root_png, pname, time, "RV", "A")
        out_rv_s_dir = os.path.join(root_png, pname, time, "RV", "S")
        out_rv_c_dir = os.path.join(root_png, pname, time, "RV", "C")


        # LA和LV\RA重合部分，LA变化
        la_a = png2mat(la_a_dir, os.listdir(la_a_dir))
        lv_a = png2mat(lv_a_dir, os.listdir(lv_a_dir))
        ra_a = png2mat(ra_a_dir, os.listdir(ra_a_dir))
        la_a_nrrd = la_a
        for index,name in enumerate(os.listdir(la_a_dir)):
            img1 = la_a[:,:,index]
            img2 = lv_a[:,:,index]
            img3 = ra_a[:,:,index]
            list = np.where(img1.astype(np.int32) + img2.astype(np.int32) == 510)
            img1[list] = 0
            list = np.where(img1.astype(np.int32) + img3.astype(np.int32) == 510)
            img1[list] = 0
            la_a[:, :, index] = img1
            # print(os.path.join(out_la_a_dir,name))
            cv2.imwrite(os.path.join(out_la_a_dir,name),img1) #保存img1，名字和原文件一样
            img1 = Inverse_ImgTrasform(img1) # 对LA重新镜像旋转变换回去再转化为nrrd
            la_a_nrrd[:,:,index]=img1
        # 转化为nrrd
        nrrd.write(os.path.join(root_nrrd, pname, time,"LA-label.nrrd"), la_a_nrrd, header=header_bg)
        # 转换为s,c
        la_a = png2mat(out_la_a_dir, os.listdir(out_la_a_dir))
        la_s = AxialToSagittal(la_a)
        mat2png(la_s, "\\" + pnumber + "_" + time + "_" + "LA" + "_S_", out_la_s_dir)
        la_c = AxialToCoronal(la_a)
        mat2png(la_c, "\\" + pnumber + "_" + time + "_" + "LA" + "_C_", out_la_c_dir)

        # RV和RA\LV重合部分，RV变化
        rv_a = png2mat(rv_a_dir, os.listdir(rv_a_dir))
        ra_a = png2mat(ra_a_dir, os.listdir(ra_a_dir))
        rv_a_nrrd = rv_a
        for index,name in enumerate(os.listdir(rv_a_dir)):
            img1 = rv_a[:,:,index]
            img2 = ra_a[:,:,index]
            list = np.where(img1.astype(np.int32) + img2.astype(np.int32) == 510)
            img1[list] = 0
            rv_a[:, :, index] = img1
            cv2.imwrite(os.path.join(out_rv_a_dir, name), img1)
            img1 = Inverse_ImgTrasform(img1)
            rv_a_nrrd[:,:,index]=img1
        # 转化为nrrd
        nrrd.write(os.path.join(root_nrrd, pname, time,"RV-label.nrrd"), rv_a_nrrd, header=header_bg)
        # 转换为s,c
        rv_a = png2mat(out_rv_a_dir, os.listdir(out_rv_a_dir))
        rv_s = AxialToSagittal(rv_a)
        mat2png(rv_s, "\\" + pnumber + "_" + time + "_" + "RV" + "_S_", out_rv_s_dir)
        rv_c = AxialToCoronal(rv_a)
        mat2png(rv_c, "\\" + pnumber + "_" + time + "_" + "RV" + "_C_", out_rv_c_dir)

