from __future__ import print_function
import argparse, os
import tensorflow as tf

from net import Net
from utils import show_all_variables
from utils import check_folder
import utils

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # change ID to designate GPU
gpu_device = '/gpu:0'

"""parsing and configuration"""


def parse_args():
    desc = "Official Tensorflow Implementation of JSI-GAN"
    parser = argparse.ArgumentParser(description=desc)

    """ Training Settings """
    parser.add_argument('--exp_num', type=int, default=1, help='The experiment number')
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test_mat', 'test_png'])
    parser.add_argument('--scale_factor', type=float, default=2, help='scale_factor for SR')
    # Set './data/train/SDR_youtube_80.mat' or './data/train/SDR_youtube_80_x4.mat' and './data/train/HDR_youtube_80.mat'
    parser.add_argument('--train_data_path_LR_SDR', type=str, default='./data/train/SDR_youtube_80.mat', help='Train input data path')
    parser.add_argument('--train_data_path_HR_HDR', type=str, default='./data/train/HDR_youtube_80.mat', help='Train GT data path')
    # For .mat file test: set './data/test/testset_SDR_x2.mat' or './data/test/testset_SDR_x4.mat' and './data/test/testset_HDR.mat'
    # For .png file test: set './data/test/PNG/SDR_x2' or './data/test/PNG/SDR_x4' and './data/test/PNG/HDR'  ** or a directory of your choice
    parser.add_argument('--test_data_path_LR_SDR', type=str, default='./data/test/testset_SDR_x2.mat', help='Test input data path')
    parser.add_argument('--test_data_path_HR_HDR', type=str, default='./data/test/testset_HDR.mat', help='Test GT data path')

    """ Directories """
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint_dir', help='checkpoint_dir path')
    parser.add_argument('--log_dir', type=str, default='logs', help='Training logs for Tensorboard')
    parser.add_argument('--test_img_dir', type=str, default='./test_img_dir', help='test_img_dir path')

    """ Hyperparameters"""
    parser.add_argument('--epoch', type=int, default=260, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=8, help='The size of batch size.')
    parser.add_argument('--val_data_size', type=int, default=500, help='Validation data size to split from train data.')
    parser.add_argument('--init_lr', type=float, default=0.0001, help='The initial learning rate')
    parser.add_argument('--lr_stair_decay_points', type=int, nargs='+', help='stair_decay - The points where the lr to be decayed (for JSInet)', default=[200, 225])
    parser.add_argument('--lr_decreasing_factor', type=float, default=0.1, help='stair_decay - lr_decreasing_factor (for JSInet)')
    parser.add_argument('--GAN_lr_linear_decay_point', type=int, default=255, help='linear_decay - lr point where linearly decreasing starts (for GAN)')

    """ Loss Coefficients """
    parser.add_argument('--rec_lambda', type=float, default=1.0, help='L2 loss lambda')
    parser.add_argument('--adv_lambda', type=float, default=1, help='GAN loss lambda')
    parser.add_argument('--fm_lambda', type=float, default=0.5, help='GAN feature matching loss lambda')
    parser.add_argument('--detail_lambda', type=float, default=0.5, help='GAN detail loss lambda')

    """ GAN Training Parameters """
    parser.add_argument('--SN_flag', type=utils.str2bool, default=True, help='Spectral Normalization (SN) flag')
    parser.add_argument('--RA_flag', type=utils.str2bool, default=True, help='Relativistic GAN flag')
    parser.add_argument('--GAN_LR_ratio', type=float, default=0.01, help='GAN learning rate ratio to inital_LR')
    parser.add_argument('--adv_weight_point', type=int, default=250, help='Epoch point where GAN loss starts to apply')

    """ Testing Settings """
    parser.add_argument('--test_patch', type=tuple, default=(6, 6), help='Divide img into patches in case of low memory')

    return check_args(parser.parse_args())


def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)
    # --log_dir
    check_folder(args.log_dir)
    # --test_img_dir
    check_folder(args.test_img_dir)

    return args


def main():
    args = parse_args()
    if args is None:
        exit()

    if args.phase == 'train':
        """ Open session """
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            model_net = Net(sess, args)
            """ Train JSInet """
            # build graph for JSInet
            model_net.build_model()
            # show network architecture
            show_all_variables()
            print("Model: JSInet")
            print("[*] JSInet training starts")
            model_net.train()
            print("[*] JSInet training finished! ")

            """ Train JSI-GAN """
            # build graph for JSI-GAN
            model_net.build_model_GAN()
            print("Model: JSI-GAN")
            print("[*] JSI-GAN training starts from pretrained_epoch_%d:" \
                  % (args.adv_weight_point))
            model_net.train_GAN()
            print("[*] JSI-GAN training finished! ")

    elif args.phase == 'test_mat':
        """ Open session """
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            model_net = Net(sess, args)
            # build graph (building GAN is not necessary for testing)
            model_net.build_model()
            # show network architecture
            show_all_variables()
            # launch the graph in a session
            print("Model: JSI-GAN")
            print("[*] Testing on .mat file starts")
            model_net.test_mat()
            print("[*] Testing finished!")

    elif args.phase == 'test_png':
        """ Open session """
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            model_net = Net(sess, args)
            # build graph (building GAN is not necessary for testing)
            model_net.build_model()
            # show network architecture
            show_all_variables()
            # launch the graph in a session
            print("Model: JSI-GAN")
            print("[*] Testing on .png file starts")
            model_net.test_png()
            print("[*] Testing finished!")


if __name__ == '__main__':
    main()
