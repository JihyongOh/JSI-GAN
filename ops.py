from utils import *


##################################################################################
# Layers
##################################################################################

# convolution layer for JSInet
def conv2d(x, shape, name):
    w = tf.get_variable(name + '/w', shape, initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    b = tf.get_variable(name + '/b', shape[3], initializer=tf.constant_initializer(0))
    n = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', name=name) + b
    return n


# convolution layer for discriminator
def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
    weight_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
    weight_regularizer = tf.contrib.layers.l2_regularizer(0.0001)
    weight_regularizer_fully = tf.contrib.layers.l2_regularizer(0.0001)
    with tf.variable_scope(scope):
        if pad > 0:
            h = x.get_shape().as_list()[1]
            if h % stride == 0:
                pad = pad * 2
            else:
                pad = max(kernel - (h % stride), 0)

            pad_top = pad // 2
            pad_bottom = pad - pad_top
            pad_left = pad // 2
            pad_right = pad - pad_left

            if pad_type == 'zero':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            if pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias)

        return x

##################################################################################
# Blocks
##################################################################################

# residual block
def res_block(x, c, name):
    with tf.variable_scope(name):
        n = conv2d(relu(x), [3, 3, c, c], 'conv/0')
        n = conv2d(relu(n), [3, 3, c, c], 'conv/1')
        n = x + n
    return n


# residual block with concat
def res_block_concat(x, c1, c, name):
    with tf.variable_scope(name):
        n = conv2d(relu(x), [3, 3, c1, c], 'conv/0')
        n = conv2d(relu(n), [3, 3, c, c], 'conv/1')
        n = x[:, :, :, :c] + n
    return n


# discriminator block
def dis_block(n, c, i, FM_list, SN_flag, is_training):
    out = conv(n, channels=c, kernel=4, stride=2, pad=1, sn=SN_flag, use_bias=False,
               scope='d_conv/' + str(2 * i + 2))
    out = lrelu(batch_norm(out, is_training, scope='d_bn/' + str(2 * i + 1)))
    FM_list.append(out) # after x2 down-sampling, append to FM_list

    out = conv(out, channels=c * 2, kernel=3, stride=1, pad=1, sn=SN_flag, use_bias=False,
               scope='d_conv/' + str(2 * i + 3))
    out = lrelu(batch_norm(out, is_training, scope='d_bn/' + str(2 * i + 2)))
    return out, FM_list

##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)

##################################################################################
# Normalization
##################################################################################

def batch_norm(x, is_training, scope):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        scale=True,
                                        is_training=is_training,
                                        scope=scope)


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

##################################################################################
# Loss function
##################################################################################

def L2_loss(x, y):
    loss = tf.reduce_mean(tf.square(x - y))
    return loss


def discriminator_loss(Ra, real, fake):
    # Hinge GAN loss
    real_loss = 0
    fake_loss = 0
    if Ra:
        real_logit = (real - tf.reduce_mean(fake))
        fake_logit = (fake - tf.reduce_mean(real))

        real_loss = tf.reduce_mean(relu(1.0 - real_logit))
        fake_loss = tf.reduce_mean(relu(1.0 + fake_logit))
    else:
        real_loss = tf.reduce_mean(relu(1.0 - real))
        fake_loss = tf.reduce_mean(relu(1.0 + fake))
    loss = real_loss + fake_loss

    return loss


def generator_loss(Ra, real, fake):
    # Hinge GAN loss
    fake_loss = 0
    real_loss = 0
    if Ra:
        fake_logit = (fake - tf.reduce_mean(real))
        real_logit = (real - tf.reduce_mean(fake))

        fake_loss = tf.reduce_mean(relu(1.0 - fake_logit))
        real_loss = tf.reduce_mean(relu(1.0 + real_logit))
    else:
        fake_loss = -tf.reduce_mean(fake)
    loss = fake_loss + real_loss

    return loss


def FM_loss(x, y, num=4):
    FM_loss = 0.0
    for i in range(num):
        FM_loss += L2_loss(x[i], y[i])

    return FM_loss

##################################################################################
# Filter
##################################################################################

# guided filter
def guidedfilter(img, r, eps):
    img2 = tf.concat([img, img * img], axis=3)
    img2 = boxfilter(img2, r)
    mean_i, mean_ii = tf.split(img2, 2, axis=3)

    var_i = mean_ii - mean_i * mean_i

    a = var_i / (var_i + eps)
    b = mean_i - a * mean_i

    ab = tf.concat([a, b], axis=3)
    ab = boxfilter(ab, r)

    mean_a, mean_b = tf.split(ab, 2, axis=3)
    q = mean_a * img + mean_b
    return q


def boxfilter(x, szf):
    y = tf.identity(x)
    szy = y.shape.as_list()

    bf = tf.ones([szf, szf, 1, 1], tf.float32) / (szf ** 2)
    bf = tf.tile(bf, [1, 1, szy[3], 1])
    pp = int((szf - 1) / 2)

    y = tf.pad(y, [[0, 0], [pp, pp], [pp, pp], [0, 0]], "REFLECT")
    y = tf.nn.depthwise_conv2d(y, bf, strides=[1, 1, 1, 1], padding='VALID')
    return y

##################################################################################
# Dynamic filter generation network components
##################################################################################

# perform separable local convolution
def dyn_sep_up_operation(x, Fv, Fh, k_sz, sf):
    '''
    Dynamic separable upsampling operation with 1D separable local kernels.
    x: [B, H, W, C], Fv: [B, H, W, 41*sf*sf], Fh: [B, H, W, 41*sf*sf]
    out: [B, H*sf, W*sf, C]
    '''
    sz = tf.shape(x)
    sz_F = tf.shape(Fv)
    pad = k_sz // 2  # local filter pad size

    out_v = tf.zeros([sz[0], sz[1], sz[2], sz[3] * sf ** 2])  # [B, H, W, C*sf*sf]
    out_h = tf.zeros([sz[0], sz[1], sz[2], sz[3] * sf ** 2])  # [B, H, W, C*sf*sf]

    img_pad = tf.pad(x, tf.constant([[0, 0], [pad, pad], [0, 0], [0, 0]]))  # [B, H+2*pad, W, C]
    img_pad_y = tf.tile(tf.expand_dims(img_pad[:, :, :, 0], axis=3), [1, 1, 1, sf ** 2])
    img_pad_u = tf.tile(tf.expand_dims(img_pad[:, :, :, 1], axis=3), [1, 1, 1, sf ** 2])
    img_pad_v = tf.tile(tf.expand_dims(img_pad[:, :, :, 2], axis=3), [1, 1, 1, sf ** 2])
    img_pad = tf.concat([img_pad_y, img_pad_u, img_pad_v], 3)   # [B, H, W, C*sf*sf]

    # vertical 1D filter
    for i in range(k_sz):
        out_v = out_v + img_pad[:, i:i + sz[1], :, :] * tf.tile(Fv[:, :, :, i:k_sz * sf ** 2:k_sz], [1, 1, 1, 3])
    img_pad = tf.pad(out_v, tf.constant([[0, 0], [0, 0], [pad, pad], [0, 0]]))  # [B, H, W+2*pad, C]
    # horizontal 1D filter
    for i in range(k_sz):
        out_h = out_h + img_pad[:, :, i:i + sz[2], :] * tf.tile(Fh[:, :, :, i:k_sz * sf ** 2:k_sz], [1, 1, 1, 3])

    # depth to space upsampling (YUV)
    out = tf.depth_to_space(out_h[:, :, :, 0:sf ** 2], sf)
    out = tf.concat([out, tf.depth_to_space(out_h[:, :, :, sf ** 2:2 * sf ** 2], sf)], 3)
    out = tf.concat([out, tf.depth_to_space(out_h[:, :, :, 2 * sf ** 2:3 * sf ** 2], sf)], 3)
    return out


def dyn_2D_up_operation(x, F, k_sz, sf=2, scope="Dynamic_2D_Upsampling"):
    '''
    Dynamic 2D upsampling operation where generated_filter is applied locally on the input
    and pixel shuffle is applied for reordering.
    x: [B, H, W, C], F: [B, H, W, 9*9*sf*sf]
    y: [B, H*sf, W*sf, 3]
    '''
    with tf.variable_scope(scope):
        y = []
        sz = tf.shape(F)  # [B, H, W, 9*9*sf*sf]
        F_new = tf.reshape(F, [sz[0], sz[1], sz[2], k_sz[0]*k_sz[1], sf ** 2])  # [B, H, W, 9*9, sf*sf]
        F_new = tf.nn.softmax(F_new, dim=3)  # softmax on each 9x9 filter
        for ch in range(3):  # loop over YUV channels
            # apply dynamic filtering operation
            temp = dyn_2D_filter(x[:, :, :, ch], F_new, k_sz)  # [B, H, W, sf*sf]
            # apply pixel shuffle for upsampling
            temp = tf.depth_to_space(temp, sf)  # [B, H*sf, W*sf, 1]
            y += [temp]
        # concat YUV channels
        y = tf.concat(y, axis=3)  # [B, H*sf, W*sf, 3]

        return y


def dyn_2D_filter(x, F, k_sz, scope="Dynamic_2D_Filtering"):
    '''
    Efficient calculation for dynamic 2D filtering operation.
    Applies the local 9x9 filters at the corresponding grid location (h, w).
    x: [B, H, W], F: [B, H, W, 9*9, sf*sf]
    y: [B, H, W, sf*sf]
    '''
    with tf.variable_scope(scope):
        # make tower
        f_localexpand_np = np.reshape(np.eye(k_sz[0]*k_sz[1], k_sz[0]*k_sz[1]), (k_sz[0], k_sz[1], 1, k_sz[0]*k_sz[1]))
        f_localexpand = tf.constant(f_localexpand_np, dtype='float32', name='filter_localexpand')  # [9, 9, 1, 81]
        # get the 9x9 neighborhood of each pixel
        x = tf.expand_dims(x, axis=3)  # [B, H, W, 1]
        x_localexpand = tf.nn.conv2d(x, f_localexpand, [1, 1, 1, 1], 'SAME') # [B, H, W, 9*9]
        x_localexpand = tf.expand_dims(x_localexpand, axis=3)  # [B, H, W, 1, 9*9]
        # apply local filters
        y = tf.matmul(x_localexpand, F)  # [B, H, W, 1, sf*sf] = [B, H, W, 1, 9*9] x [B, H, W, 9*9, sf*sf]
        y = tf.squeeze(y, axis=3)  # [B, H, W, sf*sf]
        return y

##################################################################################
# Misc
##################################################################################

# initialize the uninitialized variables
def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

        print("Success to initialize uninitialized variables.")
