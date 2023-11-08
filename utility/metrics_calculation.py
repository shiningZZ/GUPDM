# # -*- coding: utf-8 -*-
# import os
# import cv2
# import numpy as np
# from skimage.metrics import structural_similarity, peak_signal_noise_ratio
# from uiqm_utils import getUIQM



# def calculate_UIQM(image_path, resize_size=(256, 256)):
#     image_list = os.listdir(image_path)
#     uiqms = []
#     uicms = []
#     uisms = []
#     uiconms = []
#     for img in image_list:
#         image = os.path.join(image_path, img)
#
#         image = cv2.imread(image)
#         image = cv2.resize(image, resize_size)
#
#         # calculate UIQM
#         uiqm,uicm,uism,uiconm=getUIQM(image)
#         uiqms.append(uiqm)
#         uicms.append(uicm)
#         uisms.append(uism)
#         uiconms.append(uiconm)
#
#     return np.array(uiqms),np.array(uicms),np.array(uisms),np.array(uiconms)
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio,mean_squared_error
from utility.uiqm_utils import getUIQM


#最新

def calculate_metrics_ssim_psnr_all(generated_image_path, ground_truth_image_path, resize_size=(256, 256)):

    generated_image_list = os.listdir(generated_image_path)
    error_list_ssim, error_list_psnr,error_list_ssim_YCBCR, error_list_psnr_YCBCR,error_list_ssim_GRAY, error_list_psnr_GRAY,error_list_mse = [], [],[], [],[], [],[]

    for img in generated_image_list:
        label_img = img
        generated_image = os.path.join(generated_image_path, img)
        ground_truth_image = os.path.join(ground_truth_image_path, label_img)
        #print(generated_image)
        generated_image = cv2.imread(generated_image)
        generated_image = cv2.resize(generated_image, resize_size)


        ground_truth_image = cv2.imread(ground_truth_image)
        try:
            ground_truth_image = cv2.resize(ground_truth_image, resize_size)
        except:
            ground_truth_image = os.path.join(ground_truth_image_path, label_img[:-4]+'.jpg')
            ground_truth_image = cv2.imread(ground_truth_image)
            ground_truth_image = cv2.resize(ground_truth_image, resize_size)


        # calculate SSIM
        error_ssim, diff_ssim = structural_similarity(generated_image, ground_truth_image, full=True, channel_axis=-1)
        error_list_ssim.append(error_ssim)

        # generated_image = cv2.cvtColor(generated_image, cv2.COLOR_BGR2GRAY)
        # ground_truth_image = cv2.cvtColor(ground_truth_image, cv2.COLOR_BGR2GRAY)

        # calculate PSNR
        error_psnr = peak_signal_noise_ratio(generated_image, ground_truth_image)
        error_list_psnr.append(error_psnr)



        # calculate MSE
        error_mse=mean_squared_error(generated_image,ground_truth_image)
        error_list_mse.append(error_mse)




    return np.mean(np.array(error_list_ssim)), np.mean(np.array(error_list_psnr)),np.mean(np.array(error_list_mse))


















def calculate_metrics_ssim_psnr1(generated_image_path, ground_truth_image_path, resize_size=(256, 256)):
    generated_image_list = os.listdir(generated_image_path)
    error_list_ssim, error_list_psnr = [], []

    for img in generated_image_list:
        label_img = img
        generated_image = os.path.join(generated_image_path, img)
        ground_truth_image = os.path.join(ground_truth_image_path, label_img)
        generated_image = cv2.imread(generated_image)
        generated_image = cv2.resize(generated_image, resize_size)

        ground_truth_image = cv2.imread(ground_truth_image)

        ground_truth_image = cv2.resize(ground_truth_image, resize_size)

        # calculate SSIM
        error_ssim, diff_ssim = structural_similarity(generated_image, ground_truth_image, full=True, multichannel=True)
        error_list_ssim.append(error_ssim)

        generated_image = cv2.cvtColor(generated_image, cv2.COLOR_BGR2GRAY)
        ground_truth_image = cv2.cvtColor(ground_truth_image, cv2.COLOR_BGR2GRAY)
        # generated_image = cv2.cvtColor(generated_image, cv2.COLOR_BGR2RGB)
        # ground_truth_image = cv2.cvtColor(ground_truth_image, cv2.COLOR_BGR2RGB)

        # calculate PSNR
        error_psnr = peak_signal_noise_ratio(generated_image, ground_truth_image)
        error_list_psnr.append(error_psnr)

    return np.array(error_list_ssim), np.array(error_list_psnr)


def calculate_metrics_ssim_psnr(generated_image_path, ground_truth_image_path, resize_size=(256, 256)):
    generated_image_list = os.listdir(generated_image_path)
    error_list_ssim, error_list_psnr = [], []

    for img in generated_image_list:
        label_img = img
        generated_image = os.path.join(generated_image_path, img)
        ground_truth_image = os.path.join(ground_truth_image_path, label_img)
        generated_image = cv2.imread(generated_image)
        generated_image = cv2.resize(generated_image, resize_size)
        generated_image=cv2.cvtColor(cv2.cvtColor(generated_image, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2YCR_CB)

        ground_truth_image = cv2.imread(ground_truth_image)
        ground_truth_image = cv2.resize(ground_truth_image, resize_size)
        ground_truth_image= cv2.cvtColor(cv2.cvtColor(ground_truth_image, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2YCR_CB)

        # calculate SSIM
        error_ssim, diff_ssim = structural_similarity(generated_image, ground_truth_image, full=True, multichannel=True)
        error_list_ssim.append(error_ssim)

        # generated_image = cv2.cvtColor(generated_image, cv2.COLOR_BGR2GRAY)
        # ground_truth_image = cv2.cvtColor(ground_truth_image, cv2.COLOR_BGR2GRAY)

        # calculate PSNR
        error_psnr = peak_signal_noise_ratio(generated_image, ground_truth_image)
        error_list_psnr.append(error_psnr)

    return np.array(error_list_ssim), np.array(error_list_psnr)

def calculate_UIQM(image_path, resize_size=(256, 256)):
    image_list = os.listdir(image_path)
    uiqms = []
    uicms = []
    uisms = []
    uiconms = []
    for img in image_list:
        image = os.path.join(image_path, img)

        image = cv2.imread(image)
        image = cv2.resize(image, resize_size)
        #image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2YCR_CB)

        # calculate UIQM
        uiqm,uicm,uism,uiconm=getUIQM(image)
        uiqms.append(uiqm)
        uicms.append(uicm)
        uisms.append(uism)
        uiconms.append(uiconm)

    return np.array(uiqms),np.array(uicms),np.array(uisms),np.array(uiconms)



def calculate_metrics_ssim_psnr_mse(generated_image_path, ground_truth_image_path, resize_size=(256, 256)):
    generated_image_list = os.listdir(generated_image_path)
    error_list_ssim, error_list_psnr,error_list_psnr1 ,error_list_psnr2 ,error_list_mse , error_list_mse1,error_list_mse2 = [], [], [],[],[],[],[]

    for img in generated_image_list:
        label_img = img
        generated_image = os.path.join(generated_image_path, img)
        ground_truth_image = os.path.join(ground_truth_image_path, label_img)
        generated_image = cv2.imread(generated_image)
        generated_image = cv2.resize(generated_image, resize_size)

        ground_truth_image = cv2.imread(ground_truth_image)

        ground_truth_image = cv2.resize(ground_truth_image, resize_size)

        # calculate SSIM
        error_ssim, diff_ssim = structural_similarity(generated_image, ground_truth_image, full=True, multichannel=True)
        error_list_ssim.append(error_ssim)

        # generated_image = cv2.cvtColor(generated_image, cv2.COLOR_BGR2GRAY)
        # ground_truth_image = cv2.cvtColor(ground_truth_image, cv2.COLOR_BGR2GRAY)
        generated_image1 = cv2.cvtColor(generated_image, cv2.COLOR_BGR2RGB)
        ground_truth_image1 = cv2.cvtColor(ground_truth_image, cv2.COLOR_BGR2RGB)

        generated_image2 = cv2.cvtColor(generated_image, cv2.COLOR_BGR2GRAY)
        ground_truth_image2 = cv2.cvtColor(ground_truth_image, cv2.COLOR_BGR2GRAY)

        # calculate PSNR
        error_psnr = peak_signal_noise_ratio(generated_image, ground_truth_image)
        error_psnr1 = peak_signal_noise_ratio(generated_image1, ground_truth_image1)
        error_psnr2 = peak_signal_noise_ratio(generated_image2, ground_truth_image2)

        # mse
        error_mse = mean_squared_error(generated_image, ground_truth_image)
        error_mse1 = mean_squared_error(generated_image1, ground_truth_image1)
        error_mse2 = mean_squared_error(generated_image2, ground_truth_image2)



        error_list_psnr.append(error_psnr)
        error_list_psnr1.append(error_psnr1)
        error_list_psnr2.append(error_psnr2)

        error_list_mse.append(error_mse)
        error_list_mse1.append(error_mse1)
        error_list_mse2.append(error_mse2)

        #BGR,RGB,GRAY
    return np.array(error_list_ssim), np.array(error_list_psnr) ,np.array(error_list_psnr1),np.array(error_list_psnr2),np.array(error_list_mse),np.array(error_list_mse1),np.array(error_list_mse2)