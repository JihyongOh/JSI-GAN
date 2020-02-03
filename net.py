from __future__ import division
from __future__ import print_function
import glob
from datetime import datetime
import time
import math
from PIL import Image

from ops import *
import utils


class Net(object):

    def __init__(self, sess, args):
        self.sess = sess
        self.args = args
        self.model_name = "JSI-GAN"
        """ Training Settings """
        self.exp_num = args.exp_num
        self.phase = args.phase
        self.scale_factor = args.scale_factor
        self.train_data_path_LR_SDR = args.train_data_path_LR_SDR
        self.train_data_path_HR_HDR = args.train_data_path_HR_HDR
        self.test_data_path_LR_SDR = args.test_data_path_LR_SDR
        self.test_data_path_HR_HDR = args.test_data_path_HR_HDR
        """ Directories """
        self.checkpoint_dir = args.checkpoint_dir
        self.log_dir = args.log_dir
        self.test_img_dir = args.test_img_dir
        """ Hyperparameters """
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.val_data_size = args.val_data_size
        self.init_lr = args.init_lr
        self.lr_stair_decay_points = args.lr_stair_decay_points
        self.lr_decreasing_factor = args.lr_decreasing_factor
        self.GAN_lr_linear_decay_point = args.GAN_lr_linear_decay_point
        """ Loss Coefficients """
        self.rec_lambda = args.rec_lambda
        self.adv_lambda = args.adv_lambda
        self.fm_lambda = args.fm_lambda
        self.detail_lambda = args.detail_lambda
        """ GAN Training Parameters """
        self.SN_flag = args.SN_flag
        self.RA_flag = args.RA_flag
        self.GAN_LR_ratio = args.GAN_LR_ratio
        self.adv_weight_point = args.adv_weight_point
        """ Testing Settings """
        self.test_patch = args.test_patch

        """ Print all 'args' information """
        print('Model arguments, [{:s}]'.format((str(datetime.now())[:-7])))
        for arg in vars(args):
            print('# {} : {}'.format(arg, getattr(args, arg)))

    def model(self, img, sf, reuse=False, scope="model"):
        sz = img.shape
        with tf.variable_scope(scope, reuse=reuse):
            skip = dict()
            ###==================== Local Contrast Enhancement Subnet ======================###
            ch = 64
            b = guidedfilter(img, 5, 0.01)  # base layer
            with tf.variable_scope('local_contrast_enhancement'):
                n1 = conv2d(b, [3, 3, 3, ch], 'conv/0')
                for i in range(4):
                    n1 = res_block(n1, ch, 'res_block/%d' % i)
                n1 = tf.nn.relu(n1)
                # 2D local filters
                local_filter_2D = conv2d(n1, [3, 3, ch, (9 ** 2) * (sf ** 2)], 'conv_k')  # [B, H, W, (9x9)*(sfxsf)]
                # dynamic 2D upsampling with 2D local filters
                pred_C = dyn_2D_up_operation(b, local_filter_2D, [9, 9], sf, "Dynamic_2D_Upsampling")  # [B, H*sf, W*sf, 3]
                # local contrast mask
                pred_C = 2 * tf.nn.sigmoid(pred_C)

            ###==================== Detail Restoration Subnet ======================###
            ch = 64
            d = tf.div(img, b + 1e-15)  # detail layer
            with tf.variable_scope('detail_restoration'):
                n3 = conv2d(d, [3, 3, 3, ch], 'conv/0')
                for i in range(4):
                    n3 = res_block(n3, ch, 'res_block/%d' % i)
                    if i == 0:
                        d_feature = n3
                n3 = tf.nn.relu(n3)
                # separable 1D filters
                dr_k_h = conv2d(n3, [3, 3, ch, 41 * sf ** 2], 'conv_k_h')
                dr_k_v = conv2d(n3, [3, 3, ch, 41 * sf ** 2], 'conv_k_v')
                # dynamic separable upsampling with separable 1D local filters
                pred_D = dyn_sep_up_operation(d, dr_k_v, dr_k_h, 41, sf)

            ###==================== Image Reconstruction Subnet ======================###
            with tf.variable_scope('image_reconstruction'):
                n4 = conv2d(img, [3, 3, 3, ch], 'conv/0')
                for i in range(4):
                    if i == 1:
                        n4 = tf.concat([n4, d_feature], axis=3)
                        n4 = res_block_concat(n4, ch * 2, ch, 'res_block/%d' % i)
                    else:
                        n4 = res_block(n4, ch, 'res_block/%d' % i)
                n4 = tf.nn.relu(n4)

                n4 = tf.nn.relu(conv2d(n4, [3, 3, ch, ch * sf * sf], 'conv/1'))
                n4 = tf.depth_to_space(n4, sf, name='pixel_shuffle')
                pred_I = conv2d(n4, [3, 3, ch, 3], 'conv/2')

            ###======================== prediction =========================###
            pred = (pred_I + pred_D) * pred_C

            return pred

    def discriminator_FM(self, x_init, is_training=True, reuse=False, scope="discriminator_FM"):
        with tf.variable_scope(scope, reuse=reuse):
            FM_list = []
            ch = 32

            n = lrelu(conv(x_init, ch, 3, 1, 1, sn=self.SN_flag, use_bias=True, scope='d_conv/1'))
            for i in range(4):
                n, FM_list = dis_block(n, ch, i, FM_list, self.SN_flag, is_training)
                ch = ch * 2

            n = lrelu(batch_norm(conv(n, channels=ch, kernel=4, stride=2, pad=1, sn=self.SN_flag,
                                use_bias=False, scope='d_conv/10'), is_training, 'd_bn/9'))
            n = lrelu(batch_norm(conv(n, channels=ch, kernel=5, stride=1, sn=self.SN_flag,
                                use_bias=False, scope='d_conv/11'), is_training, 'd_bn/10'))
            n = batch_norm(conv(n, channels=1, kernel=1, stride=1, sn=self.SN_flag,
                          use_bias=False, scope='d_conv/12'), is_training, 'd_bn/11')

            out_logit = n
            out = tf.nn.sigmoid(out_logit)  # [B,1]

            return out, out_logit, FM_list

    def build_model(self):
        """ Read training data """
        data_path = self.train_data_path_LR_SDR
        label_path = self.train_data_path_HR_HDR

        data, label = read_mat_file(data_path, label_path, 'SDR_data', 'HDR_data')
        self.data_val = data[-self.val_data_size:, :, :, :]
        self.label_val = label[-self.val_data_size:, :, :, :]
        self.data = data[:-self.val_data_size, :, :, :]
        self.label = label[:-self.val_data_size, :, :, :]

        # calculate number of iterations
        self.data_sz = data.shape
        self.train_iter = math.floor((self.data_sz[0] - self.val_data_size) / self.batch_size)
        self.val_iter = math.floor(self.val_data_size / self.batch_size)

        """ Learning rate schedule: Stair decay """
        self.global_step = tf.Variable(initial_value=0, trainable=False)
        print("batch_size:",self.batch_size)
        self.epoch_lr_to_be_decayed_boundaries = [y * (self.train_iter) for y in
                                                  self.lr_stair_decay_points]
        self.epoch_lr_to_be_decayed_value = [self.init_lr * (self.lr_decreasing_factor ** y) for y in
                                             range(len(self.lr_stair_decay_points) + 1)]
        self.lr = tf.train.piecewise_constant(self.global_step, self.epoch_lr_to_be_decayed_boundaries,
                                              self.epoch_lr_to_be_decayed_value)
        print("lr_type: stair_decay")

        """ Definie Model for JSInet """
        # define variables for data
        self.input_ph = tf.placeholder(tf.float32, shape=(None, self.data_sz[1], self.data_sz[2], self.data_sz[3]))
        # define variables for label
        self.label_ph = tf.placeholder(tf.float32, shape=(None, self.data_sz[1]*self.scale_factor, self.data_sz[2]*self.scale_factor, self.data_sz[3]))
        # network model
        self.pred = self.model(self.input_ph, self.scale_factor, reuse=False, scope='Network')
        """ Define Loss """
        self.rec_loss = L2_loss(self.pred, self.label_ph)
        self.train_PSNR = tf.reduce_mean(tf.image.psnr(self.pred, self.label_ph, max_val=1.0))
        """ Optimizer """
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optim =  tf.train.AdamOptimizer(self.lr).minimize(self.rec_loss, global_step=self.global_step)
        """" Summary """
        self.rec_loss_sum = tf.summary.scalar("rec_loss", self.rec_loss)
        self.train_PSNR_sum = tf.summary.scalar("train_PSNR", self.train_PSNR)
        self.total_summary_loss = tf.summary.merge([self.rec_loss_sum, self.train_PSNR_sum])

        """ For Testing Phase """
        if self.phase == 'test_mat' or self.phase == 'test_png':
            self.test_input_ph = tf.placeholder(tf.float32, shape=(1, None, None, self.data_sz[3]))
            self.test_pred = self.model(self.test_input_ph, self.scale_factor, reuse=True, scope='Network')

    def build_model_GAN(self):
        """ Learning rate schedule: Linear decay """ #
        self.lr_GAN = tf.placeholder(tf.float32, name='learning_rate')
        print("lr_type: linear_decay")

        """ Define Discriminator """
        # output of D for real images
        D_real, D_real_logits, D_real_FM_list = self.discriminator_FM(self.label_ph, is_training=True,
                                                   reuse=False, scope="Discriminator_FM")
        # output of D for fake images
        D_fake, D_fake_logits,D_fake_FM_list = self.discriminator_FM(self.pred, is_training=True, reuse=True,
                                                   scope="Discriminator_FM")

        """ Define Detail Discriminator """
        # compute the detail layers for the dicriminator (reuse)
        base_GT = guidedfilter(self.label_ph, 5, 0.01)
        self.detail_GT = tf.div(self.label_ph, base_GT + 1e-15)
        base_pred = guidedfilter(self.pred, 5, 0.01)
        self.detail_pred = tf.div(self.pred, base_pred + 1e-15)

        # detail layer output of D for real images
        D_detail_real, D_detail_real_logits, D_detail_real_FM_list = \
            self.discriminator_FM(self.detail_GT, is_training=True, reuse=False, scope="Discriminator_Detail")
        # detail layer output of D for fake images
        D_detail_fake, D_detail_fake_logits, D_detail_fake_FM_list = \
            self.discriminator_FM(self.detail_pred, is_training=True, reuse=True, scope="Discriminator_Detail")

        """ Loss """
        # original GAN (hinge GAN)
        self.d_adv_loss = discriminator_loss(Ra=self.RA_flag, real=D_real_logits, fake=D_fake_logits)
        self.g_adv_loss = generator_loss(Ra=self.RA_flag, real=D_real_logits,
                                         fake=D_fake_logits)
        # detail GAN (hinge GAN)
        self.d_detail_adv_loss = self.detail_lambda * \
                                 discriminator_loss(Ra=self.RA_flag, real=D_detail_real_logits, fake=D_detail_fake_logits)
        self.g_detail_adv_loss = self.detail_lambda * \
                                 generator_loss(Ra=self.RA_flag, real=D_detail_real_logits, fake=D_detail_fake_logits)
        # feature matching (FM) loss
        self.FM_loss = FM_loss(D_real_FM_list, D_fake_FM_list, 4)
        self.FM_detail_loss = self.detail_lambda * FM_loss(D_detail_real_FM_list, D_detail_fake_FM_list, 4)
        """ Final Losses """
        self.d_final_FM_loss = self.d_adv_loss
        self.d_final_detail_loss = self.d_detail_adv_loss
        self.g_final_loss = self.rec_lambda * self.rec_loss + self.adv_lambda * (self.g_adv_loss + self.g_detail_adv_loss) \
                             + self.fm_lambda * (self.FM_loss + self.FM_detail_loss)

        """ Optimizers for GAN """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_FM_vars = [var for var in t_vars if 'Discriminator_FM' in var.name]
        d_detail_vars = [var for var in t_vars if 'Discriminator_Detail' in var.name]
        g_vars = [var for var in t_vars if 'Network' in var.name] # generator

        with tf.variable_scope("Include_DetailGAN"):
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.d_FM_optim = tf.train.AdamOptimizer(self.lr * self.GAN_LR_ratio, beta1=0.9) \
                    .minimize(self.d_final_FM_loss, var_list=d_FM_vars)
                self.d_detail_optim = tf.train.AdamOptimizer(self.lr * self.GAN_LR_ratio, beta1=0.9) \
                    .minimize(self.d_final_detail_loss, var_list=d_detail_vars)
                self.g_optim = tf.train.AdamOptimizer(self.lr * self.GAN_LR_ratio, beta1=0.9) \
                    .minimize(self.g_final_loss, var_list=g_vars, global_step=self.global_step)

        """ Summary """
        # generator
        self.rec_loss_sum = tf.summary.scalar("rec_loss", self.rec_loss)
        self.g_adv_loss_sum = tf.summary.scalar("g_adv_loss", self.g_adv_loss)
        self.g_detail_adv_loss_sum = tf.summary.scalar("g_detail_adv_loss", self.g_detail_adv_loss)
        self.FM_loss_sum = tf.summary.scalar("FM_loss", self.FM_loss)
        self.FM_detail_loss_sum = tf.summary.scalar("FM_detail_loss", self.FM_detail_loss)
        # discriminator
        self.d_adv_loss_sum = tf.summary.scalar("d_adv_loss", self.d_adv_loss)
        self.d_detail_adv_loss_sum = tf.summary.scalar("d_detail_adv_loss", self.d_detail_adv_loss)
        self.train_PSNR_sum = tf.summary.scalar("train_PSNR", self.train_PSNR)
        # final
        self.d_final_FM_loss_sum = tf.summary.scalar("d_final_FM_loss", self.d_final_FM_loss)
        self.d_final_detail_loss_sum = tf.summary.scalar("d_final_detail_loss", self.d_final_detail_loss)
        self.g_final_loss_sum = tf.summary.scalar("g_final_loss", self.g_final_loss)
        # merge
        self.g_summary_loss = tf.summary.merge([self.rec_loss_sum, self.g_adv_loss_sum,
                                                self.g_detail_adv_loss_sum, self.FM_loss_sum,
                                                self.FM_detail_loss_sum])
        self.d_summary_loss = tf.summary.merge([self.d_adv_loss_sum, self.d_detail_adv_loss_sum,self.train_PSNR_sum])
        self.final_summary_loss = tf.summary.merge([self.d_final_FM_loss_sum, self.d_final_detail_loss_sum, self.g_final_loss_sum])

    def train(self):
        """ Initialize """  #
        tf.global_variables_initializer().run()
        # saver to save model
        self.saver = tf.train.Saver(max_to_keep=1)

        """ Summary Writer """
        summary_dir = os.path.join(self.log_dir, self.model_dir)
        # summary_dir = os.path.abspath(os.path.join(self.log_dir, self.model_dir))
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        self.writer = tf.summary.FileWriter(summary_dir, self.sess.graph)

        """ Restore Checkpoint """
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / (self.train_iter))
            start_batch_id = checkpoint_counter - start_epoch * (self.train_iter)
            self.counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            self.counter = 1
            print(" [!] Load failed...")

        """ Training """
        start_time = time.time()
        for epoch in range(start_epoch, self.adv_weight_point):
            loss_rec_list_for_epoch = []
            # shuffle index
            rand_idx = np.random.permutation(self.data_sz[0] - self.val_data_size)
            for idx in range(self.train_iter):
                data_batch = self.data[rand_idx[self.batch_size*idx:self.batch_size*(idx+1)], :, :, :]
                label_batch = self.label[rand_idx[self.batch_size*idx:self.batch_size*(idx+1)], :, :, :]
                _, total_summary_loss_str, rec_loss, lr_per_epoch, train_PSNR= self.sess.run(
                    [self.optim,
                     self.total_summary_loss,
                     self.rec_loss,
                     self.lr, self.train_PSNR], feed_dict={self.input_ph: data_batch, self.label_ph: label_batch})
                # add summary
                self.writer.add_summary(total_summary_loss_str, self.counter)
                self.counter += 1
                # append loss
                loss_rec_list_for_epoch.append(rec_loss)
                print(
                    "(per batch) Epoch: [%4d], [%4d/%4d]-th batch, time: %4.4f(minutes), "
                    "rec_loss: %.8f, train_PSNR: %.8f" \
                    % (epoch, idx, self.train_iter, (time.time() - start_time) / 60, rec_loss, train_PSNR))

            rec_loss_per_epoch = np.mean(loss_rec_list_for_epoch)
            print(
                "######### One epoch ends (average), Learning rate: %10.10f, Epoch: [%4d/%4d]-th epoch, time: %4.4f(minutes), "
                "rec_loss: %.8f  #########" \
                % (lr_per_epoch, epoch, self.epoch, (time.time() - start_time) / 60, rec_loss_per_epoch))

            """ Validation """
            val_loss_rec_list_for_epoch = []
            val_loss_PSNR_list_for_epoch = []
            for val_idx in range(self.val_iter):
                data_batch_val = self.data_val[self.batch_size*val_idx:self.batch_size*(val_idx+1), :, :, :]
                label_batch_val = self.label_val[self.batch_size*val_idx:self.batch_size*(val_idx+1), :, :, :]
                val_rec_loss, val_PSNR = self.sess.run([self.rec_loss, self.train_PSNR],
                                                       feed_dict={self.input_ph: data_batch_val,
                                                                  self.label_ph: label_batch_val})
                val_loss_rec_list_for_epoch.append(val_rec_loss)
                val_loss_PSNR_list_for_epoch.append(val_PSNR)
            val_rec_loss_per_epoch = np.mean(val_loss_rec_list_for_epoch)
            val_PSNR_per_epoch = np.mean(val_loss_PSNR_list_for_epoch)

            print(
                "######### Validation (average),Epoch: [%4d/%4d]-th epoch, time: %4.4f(minutes), val_PSNR: %.8f[dB], "
                "rec_loss: %.8f  #########" \
                % (epoch, self.epoch, (time.time() - start_time) / 60,
                   val_PSNR_per_epoch, val_rec_loss_per_epoch))

            """ Save Model """
            self.save_checkpoint(self.checkpoint_dir, 'JSInet', self.global_step.eval())
            start_batch_id = 0

    def train_GAN(self):
        """ Initialize """  #
        # load pretrained model JSInet
        self.load_pretrained_model(self.checkpoint_dir, 'JSInet')
        self.saver = tf.train.Saver(max_to_keep=1)
        # initialize uninitialized variables
        initialize_uninitialized(self.sess)
        # set start epoch
        start_epoch = self.adv_weight_point
        start_batch_id = 0

        """ Training """
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):
            rec_loss_list_for_epoch = []
            train_PSNR_list_for_epoch = []
            g_list_for_epoch = []
            g_adv_list_for_epoch = []
            g_detail_list_for_epoch = []
            d_adv_list_for_epoch = []
            d_detail_list_for_epoch = []
            FM_list_for_epoch = []
            FM_detail_list_for_epoch = []

            # shuffle index
            rand_idx = np.random.permutation(self.data_sz[0] - self.val_data_size)
            for idx in range(self.train_iter):
                data_batch = self.data[rand_idx[self.batch_size * idx:self.batch_size * (idx + 1)], :, :, :]
                label_batch = self.label[rand_idx[self.batch_size * idx:self.batch_size * (idx + 1)], :, :, :]
                # linear decay learning rate
                lr = self.init_lr if epoch < self.GAN_lr_linear_decay_point \
                    else self.init_lr * (self.epoch - epoch) / (self.epoch - self.GAN_lr_linear_decay_point)
                feed_dict = {self.lr: lr, self.input_ph: data_batch, self.label_ph: label_batch}

                """ Update 2 Discriminators """
                _, _, summary_str_d, d_adv_loss, d_detail_adv_loss = self.sess.run(
                    [self.d_FM_optim, self.d_detail_optim, self.d_summary_loss,
                     self.d_adv_loss, self.d_detail_adv_loss], feed_dict=feed_dict)
                self.writer.add_summary(summary_str_d, self.counter)

                """ Update Generator """
                _, summary_str_final_loss, summary_str_g, g_loss, rec_loss, g_adv_loss, g_detail_adv_loss, \
                lr_per_epoch, train_PSNR, FM_loss, FM_detail_loss = self.sess.run(
                    [self.g_optim, self.final_summary_loss, self.g_summary_loss,
                     self.g_final_loss, self.rec_loss, self.g_adv_loss, self.g_detail_adv_loss,
                     self.lr, self.train_PSNR, self.FM_loss, self.FM_detail_loss], feed_dict=feed_dict)
                self.writer.add_summary(summary_str_g, self.counter)
                self.writer.add_summary(summary_str_final_loss, self.counter)
                print("Epoch: [%3d], [%4d/%4d]-th batch, time: %4.2f(min.), "
                    "train_PSNR: %.3f, rec_loss: %.6f, g_loss: %.6f, g_adv_loss: %.6f, "
                    "g_detail_adv_loss: %.6f, d_adv_loss: %.6f, d_detail_adv_loss: %.6f,"
                    "FM_loss: %.6f,FM_detail_loss: %.6f," \
                    % (epoch, idx, self.train_iter, (time.time() - start_time) / 60,
                       train_PSNR, rec_loss, g_loss, g_adv_loss,
                       g_detail_adv_loss, d_adv_loss, d_detail_adv_loss,
                       FM_loss, FM_detail_loss))
                self.counter += 1

                train_PSNR_list_for_epoch.append(train_PSNR)
                rec_loss_list_for_epoch.append(rec_loss)
                g_list_for_epoch.append(g_loss)
                g_adv_list_for_epoch.append(g_adv_loss)
                g_detail_list_for_epoch.append(g_detail_adv_loss)
                d_adv_list_for_epoch.append(d_adv_loss)
                d_detail_list_for_epoch.append(d_detail_adv_loss)
                FM_list_for_epoch.append(FM_loss)
                FM_detail_list_for_epoch.append(FM_detail_loss)

            train_PSNR_for_epoch = np.mean(train_PSNR_list_for_epoch)
            rec_loss_per_epoch = np.mean(rec_loss_list_for_epoch)
            g_for_epoch = np.mean(g_list_for_epoch)
            g_adv_for_epoch = np.mean(g_adv_list_for_epoch)
            g_detail_per_epoch = np.mean(g_detail_list_for_epoch)
            d_adv_per_epoch = np.mean(d_adv_list_for_epoch)
            d_detail_per_epoch = np.mean(d_detail_list_for_epoch)
            FM_per_epoch = np.mean(FM_list_for_epoch)
            FM_detail_per_epoch = np.mean(FM_detail_list_for_epoch)

            print(
                "# (average) Epoch: [%4d], LR: %10.10f, time: %4.2f(minutes), "
                "train_PSNR: %.3f, rec_loss: %.6f, g_loss: %.6f, g_adv_loss: %.6f, g_detail_adv_loss: %.6f, "
                "d_adv_loss: %.6f, d_detail_adv_loss: %.6f," \
                % (epoch, lr_per_epoch, (time.time() - start_time) / 60,
                   train_PSNR_for_epoch, rec_loss_per_epoch, g_for_epoch, g_adv_for_epoch, g_detail_per_epoch,
                   d_adv_per_epoch, d_detail_per_epoch
                   ))
            print(
                "FM_loss: %.6f,FM_detail_loss: %.6f,"\
                % (FM_per_epoch, FM_detail_per_epoch))

            """ Validation """
            val_loss_rec_list_for_epoch = []
            val_loss_PSNR_list_for_epoch = []
            for val_idx in range(self.val_iter):
                data_batch_val = self.data_val[self.batch_size * val_idx:self.batch_size * (val_idx + 1), :, :, :]
                label_batch_val = self.label_val[self.batch_size * val_idx:self.batch_size * (val_idx + 1), :, :, :]

                val_rec_loss, val_PSNR = self.sess.run([self.rec_loss, self.train_PSNR],
                    feed_dict={self.input_ph: data_batch_val, self.label_ph: label_batch_val})

                val_loss_rec_list_for_epoch.append(val_rec_loss)
                val_loss_PSNR_list_for_epoch.append(val_PSNR)

            val_rec_loss_per_epoch = np.mean(val_loss_rec_list_for_epoch)
            val_PSNR_per_epoch = np.mean(val_loss_PSNR_list_for_epoch)

            print(
                "######### Validation (average),Epoch: [%4d/%4d]-th epoch, time: %4.4f(minutes), val_PSNR: %.3f[dB], "
                "rec_loss: %.6f  #########" \
                % (epoch, self.epoch, (time.time() - start_time) / 60, val_PSNR_per_epoch, val_rec_loss_per_epoch))

            """ Save model """
            self.save_checkpoint(self.checkpoint_dir, 'JSI-GAN', self.global_step.eval())
            start_batch_id = 0

    def test_mat(self):
        # saver to save model
        self.saver = tf.train.Saver()
        tf.global_variables_initializer().run()
        # restore check-point
        self.load(self.checkpoint_dir)  # for testing JSI-GAN
        # self.load_pretrained_model(self.checkpoint_dir, 'JSInet')  # for testing JSInet

        """" Test """
        """ Matlab data for test """
        data_path_test = self.test_data_path_LR_SDR
        label_path_test = self.test_data_path_HR_HDR
        data_test, label_test = read_mat_file(data_path_test, label_path_test, 'SDR_YUV', 'HDR_YUV')
        data_sz = data_test.shape
        label_sz = label_test.shape

        """ Make "test_img_dir" per experiment """
        test_img_dir = os.path.join(self.test_img_dir, self.model_dir)
        if not os.path.exists(test_img_dir):
            os.makedirs(test_img_dir)

        """ Testing """
        patch_boundary = 10  # set patch boundary to reduce edge effect around patch edges
        test_loss_PSNR_list_for_epoch = []
        inf_time = []
        start_time = time.time()
        test_pred_full = np.zeros((label_sz[1], label_sz[2], label_sz[3]))
        for index in range(data_sz[0]):
            ###======== Divide Into Patches ========###
            for p in range(self.test_patch[0] * self.test_patch[1]):
                pH = p // self.test_patch[1]
                pW = p % self.test_patch[1]
                sH = data_sz[1] // self.test_patch[0]
                sW = data_sz[2] // self.test_patch[1]
                # process data considering patch boundary
                H_low_ind, H_high_ind, W_low_ind, W_high_ind = \
                    get_HW_boundary(patch_boundary, data_sz[1], data_sz[2], pH, sH, pW, sW)
                data_test_p = data_test[index, H_low_ind: H_high_ind, W_low_ind: W_high_ind, :]
                data_test_p = np.expand_dims(data_test_p, axis=0)
                ###======== Run Session ========###
                st = time.time()
                test_pred_o = self.sess.run(self.test_pred, feed_dict={self.test_input_ph: data_test_p})
                inf_time.append(time.time() - st)
                # trim patch boundary
                test_pred_t = trim_patch_boundary(test_pred_o, patch_boundary, data_sz[1], data_sz[2], pH, sH, pW, sW, self.scale_factor)
                # store in pred_full
                test_pred_full[pH * sH * self.scale_factor: (pH + 1) * sH * self.scale_factor,
                pW * sW * self.scale_factor: (pW + 1) * sW * self.scale_factor, :] = np.squeeze(test_pred_t)
            ###======== Compute PSNR & Print Results========###
            test_GT = np.squeeze(label_test[index, :, :, :])
            test_PSNR = utils.compute_psnr(test_pred_full, test_GT, 1.)
            test_loss_PSNR_list_for_epoch.append(test_PSNR)
            print(" <Test> [%4d/%4d]-th images, time: %4.4f(minutes), test_PSNR: %.8f[dB]  "
                  % (int(index), int(data_sz[0]), (time.time() - start_time) / 60, test_PSNR))

            ###======== Save Predictions as Images ========###
            utils.save_results_yuv(test_pred_full, index, test_img_dir)  # comment for faster testing
        test_PSNR_per_epoch = np.mean(test_loss_PSNR_list_for_epoch)

        print("######### Average Test PSNR: %.8f[dB]  #########" % (test_PSNR_per_epoch))
        print("######### Estimated Inference Time (per 4K frame): %.8f[s]  #########" % (np.mean(inf_time)*self.test_patch[0]*self.test_patch[1]))

    def test_png(self):
        # saver to save model
        self.saver = tf.train.Saver()
        tf.global_variables_initializer().run()
        # restore check-point
        self.load(self.checkpoint_dir)  # for testing JSI-GAN
        # self.load_pretrained_model(self.checkpoint_dir, 'JSInet')  # for testing JSInet

        """" Test """
        data_path_test = glob.glob(os.path.join(self.test_data_path_LR_SDR, '*.png'))
        label_path_test = glob.glob(os.path.join(self.test_data_path_HR_HDR, '*.png'))

        """ Make "test_img_dir" per experiment """
        test_img_dir = os.path.join(self.test_img_dir, self.model_dir)
        if not os.path.exists(test_img_dir):
            os.makedirs(test_img_dir)
        """ Testing """
        patch_boundary = 10  # set patch boundary to reduce edge effect around patch edges
        test_loss_PSNR_list_for_epoch = []
        inf_time = []
        start_time = time.time()
        for index in range(len(data_path_test)//3):
            ###======== Read Data ========###
            y = np.array(Image.open(data_path_test[3*index+2]))
            u = np.array(Image.open(data_path_test[3*index]))
            v = np.array(Image.open(data_path_test[3*index+1]))
            ###======== Pre-process Data ========###
            img = np.expand_dims(np.stack([y, u, v], axis=2), axis=0)
            data_sz = img.shape
            test_pred_full = np.zeros((data_sz[1]*self.scale_factor, data_sz[2]*self.scale_factor, data_sz[3]))
            img = np.array(img, dtype=np.double) / 255.
            data_test = np.clip(img, 0, 1)
            ###======== Divide Into Patches ========###
            for p in range(self.test_patch[0] * self.test_patch[1]):
                pH = p // self.test_patch[1]
                pW = p % self.test_patch[1]
                sH = data_sz[1] // self.test_patch[0]
                sW = data_sz[2] // self.test_patch[1]
                # process data considering patch boundary
                H_low_ind, H_high_ind, W_low_ind, W_high_ind = \
                    get_HW_boundary(patch_boundary, data_sz[1], data_sz[2], pH, sH, pW, sW)
                data_test_p = data_test[:, H_low_ind: H_high_ind, W_low_ind: W_high_ind, :]
                ###======== Run Session ========###
                st = time.time()
                test_pred_o = self.sess.run(self.test_pred, feed_dict={self.test_input_ph: data_test_p})
                inf_time.append(time.time() - st)
                # trim patch boundary
                test_pred_t = trim_patch_boundary(test_pred_o, patch_boundary, data_sz[1], data_sz[2], pH, sH, pW, sW, self.scale_factor)
                # store in pred_full
                test_pred_full[pH * sH * self.scale_factor: (pH + 1) * sH * self.scale_factor,
                pW * sW * self.scale_factor: (pW + 1) * sW * self.scale_factor, :] = np.squeeze(test_pred_t)
            ###======== Compute PSNR & Print Results========###
            label_y = np.array(Image.open(label_path_test[3*index+2]))
            label_u = np.array(Image.open(label_path_test[3*index]))
            label_v = np.array(Image.open(label_path_test[3*index+1]))
            test_GT = np.stack([label_y, label_u, label_v], axis=2)
            test_GT = np.array(test_GT, dtype=np.double) / 1023.
            test_GT = np.clip(test_GT, 0, 1)
            test_PSNR = utils.compute_psnr(test_pred_full, test_GT, 1.)
            test_loss_PSNR_list_for_epoch.append(test_PSNR)
            print(" <Test> [%4d/%4d]-th images, time: %4.4f(minutes), test_PSNR: %.8f[dB]  "
                  % (int(index), int(len(data_path_test)//3), (time.time() - start_time) / 60, test_PSNR))

            ###======== Save Predictions as Images ========###
            utils.save_results_yuv(test_pred_full, index, test_img_dir)  # comment for faster testing
        test_PSNR_per_epoch = np.mean(test_loss_PSNR_list_for_epoch)

        print("######### Average Test PSNR: %.8f[dB]  #########" % (test_PSNR_per_epoch))
        print("######### Estimated Inference Time (per 4K frame): %.8f[s]  #########" % (np.mean(inf_time)*self.test_patch[0]*self.test_patch[1]))

    @property
    def model_dir(self):
        return "{}_x{}_exp{}".format(self.model_name, self.scale_factor, self.exp_num)

    def save_checkpoint(self, checkpoint_dir, name, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, name), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def load_pretrained_model(self, checkpoint_dir, name):
        print(" [*] Reading pretrained_model checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        ckpt_numbering = self.adv_weight_point*self.train_iter
        self.saver.restore(self.sess, (os.path.join(checkpoint_dir, name+'-'+str(ckpt_numbering))))
        print(" [*] Success to read pretrained epoch {}".format(self.adv_weight_point))
