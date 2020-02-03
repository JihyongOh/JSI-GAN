from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim

import imageio
import numpy as np
import os
import h5py


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def str2bool(x):
    return x.lower() in ('true')


def compute_psnr(img_orig, img_out, peak):
    mse = np.mean(np.square(img_orig - img_out))
    psnr = 10 * np.log10(peak*peak / mse)
    return psnr


def read_mat_file(data_fname, label_fname, data_name, label_name):
    # read training data (.mat file)
    data_file = h5py.File(data_fname, 'r')
    label_file = h5py.File(label_fname, 'r')
    data = data_file[data_name][()]
    label = label_file[label_name][()]

    # change type & reorder
    data = np.array(data, dtype=np.float32) / 255.
    label = np.array(label, dtype=np.float32) / 1023.
    data = np.swapaxes(data, 1, 3)
    label = np.swapaxes(label, 1, 3)
    print('[*] Success to read .mat file')
    return data, label


def get_HW_boundary(patch_boundary, h, w, pH, sH, pW, sW):
    H_low_ind = max(pH * sH - patch_boundary, 0)
    H_high_ind = min((pH + 1) * sH + patch_boundary, h)
    W_low_ind = max(pW * sW - patch_boundary, 0)
    W_high_ind = min((pW + 1) * sW + patch_boundary, w)

    return H_low_ind, H_high_ind, W_low_ind, W_high_ind


def trim_patch_boundary(img, patch_boundary, h, w, pH, sH, pW, sW, sf):
    if patch_boundary == 0:
        img = img
    else:
        if pH * sH < patch_boundary:
            img = img
        else:
            img = img[:, patch_boundary*sf:, :, :]
        if (pH + 1) * sH + patch_boundary > h:
            img = img
        else:
            img = img[:, :-patch_boundary*sf, :, :]
        if pW * sW < patch_boundary:
            img = img
        else:
            img = img[:, :, patch_boundary*sf:, :]
        if (pW + 1) * sW + patch_boundary > w:
            img = img
        else:
            img = img[:, :, :-patch_boundary*sf, :]

    return img


def save_results_yuv(pred, index, test_img_dir):
    test_pred = np.squeeze(pred)
    test_pred = np.clip(test_pred, 0, 1) * 1023
    test_pred = np.uint16(test_pred)

    # split image
    pred_y = test_pred[:, :, 0]
    pred_u = test_pred[:, :, 1]
    pred_v = test_pred[:, :, 2]

    # save prediction - must be saved in separate channels due to 16-bit pixel depth
    imageio.imwrite(os.path.join(test_img_dir, "{}-y_pred.png".format(str(int(index) + 1).zfill(2))),
                    pred_y)
    imageio.imwrite(os.path.join(test_img_dir, "{}-u_pred.png".format(str(int(index) + 1).zfill(2))),
                    pred_u)
    imageio.imwrite(os.path.join(test_img_dir, "{}-v_pred.png".format(str(int(index) + 1).zfill(2))),
                    pred_v)
