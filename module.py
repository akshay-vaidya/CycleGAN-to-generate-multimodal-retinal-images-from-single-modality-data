from __future__ import division
from keras.layers import Conv2D,UpSampling2D
from keras.losses import binary_crossentropy
from keras.activations import sigmoid
from keras.layers.advanced_activations import LeakyReLU
#from keras.layers.normalization import InstanceNormalization
from keras.layers import Input, Lambda, add, Concatenate, Reshape, Flatten, Dropout, Multiply, Dense, Activation
from keras.models import Model
import keras.backend as k
import tensorflow as tf
from keras.initializers import RandomNormal

from keras.engine.topology import Layer
import keras.backend as k
import tensorflow as tf 

class InstanceNormalization(Layer):
    def __init__(self,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 epsilon=1e-3,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        if k.image_data_format() is 'channels_first':
            self.axis = 1
        else: # image channels x.shape[3]
            self.axis = 3
        self.epsilon = epsilon
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(input_shape[self.axis],),
                                     initializer=self.gamma_initializer,
                                     trainable=True,
                                     name='gamma')
        self.beta = self.add_weight(shape=(input_shape[self.axis],),
                                    initializer=self.beta_initializer,
                                    trainable=True,
                                    name='beta')
        super(InstanceNormalization, self).build(input_shape)

    def call(self, x):
        # spatial dimensions of input
        if k.image_data_format() is 'channels_first':
            x_w, x_h = (2, 3)
        else:
            x_w, x_h = (1, 2)

        # Very similar to batchnorm, but normalization over individual inputs.

        hw = k.cast(k.shape(x)[x_h]* k.shape(x)[x_w], k.floatx())

        # Instance means
        mu = k.sum(x, axis=x_w)
        mu = k.sum(mu, axis=x_h)
        mu = mu / hw
        mu = k.reshape(mu, (k.shape(mu)[0], k.shape(mu)[1], 1, 1))

        # Instance variences
        sig2 = k.square(x - mu)
        sig2 = k.sum(sig2, axis=x_w)
        sig2 = k.sum(sig2, axis=x_h)
        sig2 = k.reshape(sig2, (k.shape(sig2)[0], k.shape(sig2)[1], 1, 1))

        # Normalize
        y = (x - mu) / k.sqrt(sig2 + self.epsilon)

        # Scale and Shift
        if k.image_data_format() is 'channels_first':
            gamma = k.reshape(self.gamma, (1, k.shape(self.gamma)[0], 1, 1))
            beta = k.reshape(self.beta, (1, k.shape(self.beta)[0], 1, 1))
        else:
            gamma = k.reshape(self.gamma, (1, 1, 1, k.shape(self.gamma)[0]))
            beta = k.reshape(self.beta, (1, 1, 1, k.shape(self.beta)[0]))
        return gamma * y + beta


def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = k.sum(k.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / k.sqrt(s_squared_norm + k.epsilon())
    return scale * vectors

def conv2d(c_i, filters, ks=4,s=2, padding='SAME', activation=None):
    c = Conv2D(
        filters,
        kernel_size=ks,
        strides=s,
        padding=padding,
        activation=activation,
        kernel_initializer= RandomNormal(mean = 0.0, stddev= 0.02))(c_i)
    return c

def trans_conv2d(tc_i, filters, ks=4,s=2, activation=None, padding='SAME'):
    #t1 = Conv2DTranspose(filters, kernel_size=ks, strides=s, padding=padding,
    #                             activation=activation, kernel_initializer= RandomNormal(mean = 0.0, stddev= 0.02))(tc_i)
    t = UpSampling2D(size=4)(tc_i)
    t = Lambda(lambda x: tf.pad(x, [[0,0],[1,1],[1,1],[0,0]],'REFLECT'))(t)
    t1 = Conv2D(
        filters,
        kernel_size=ks,
        strides=s,
        padding=padding,
        activation=activation,
        kernel_initializer= RandomNormal(mean = 0.0, stddev= 0.02))(t)

    return t1

def residule_block(r_i, layer_output, ks=3, s=1):
    r = Lambda(lambda x: tf.pad(x, [[0,0],[1,1],[1,1],[0,0]],'REFLECT'))(r_i)
    #r = ReflectionPadding2D(padding=(1,1))(r_i)
    r = conv2d(r,layer_output,ks,s,padding= 'VALID')
    r = InstanceNormalization()(r)
    r = Lambda(lambda x: tf.pad(x, [[0,0],[1,1],[1,1],[0,0]],'REFLECT'))(r)
    #r = ReflectionPadding2D(padding=(1,1))(r)
    r = conv2d(r,layer_output,ks,s,padding= 'VALID')
    #r = InstanceNormalization()(r)
    return add([r_i , r])

def discriminator(opt):
    img = Input(shape=(opt.data_pix_size,opt.data_pix_size,opt.in_dim,))
    d1 = LeakyReLU(alpha=0.2)(conv2d(img, opt.d_fir_dim, 4, 2))
    d2 = LeakyReLU(alpha=0.2)((conv2d(d1, opt.d_fir_dim*2, 4, 2)))
    d3 = LeakyReLU(alpha=0.2)((conv2d(d2, opt.d_fir_dim*4, 4, 2)))
    d4 = LeakyReLU(alpha=0.2)((conv2d(d3, opt.d_fir_dim*8, 4, 2)))
    d5 = conv2d(d4, 1, s=1)
    return Model(inputs = img, outputs = d5)

def capsule_discriminator(opt):
    img = Input(shape=(opt.data_pix_size,opt.data_pix_size,opt.in_dim,))
    d1 = LeakyReLU(alpha=0.2)(conv2d(img, opt.d_fir_dim * 4, 4, 2))
    d2 = conv2d(d1, opt.d_fir_dim * 4, 4, 2, padding='valid')
    d3 = Reshape(target_shape=[-1, 4])(d2)
    d4 = Lambda(squash)(d3)
    d5 = Flatten()(d4)
    uhat = Dense(64, kernel_initializer='he_normal', bias_initializer='zeros', name='uhat_digitcaps')(d5)
    c = Activation(tf.nn.softmax, name='softmax_digitcaps1')(uhat)
    c = Dense(64)(c)
    x = Multiply()([uhat, c])
    s_j = LeakyReLU()(x)
    c = Activation(tf.nn.softmax, name='softmax_digitcaps2')(s_j)
    c = Dense(64)(c)
    x = Multiply()([uhat, c])
    s_j = LeakyReLU()(x)
    c = Activation(tf.nn.softmax, name='softmax_digitcaps3')(s_j)
    c = Dense(64)(c) # compute s_j
    x = Multiply()([uhat, c])
    s_j = LeakyReLU()(x)

    pred = Dense(1, activation='sigmoid')(s_j)

    
    return Model(inputs = img, outputs = pred)
    


def generator_resnet(opt):
    img = Input(shape=(opt.data_pix_size,opt.data_pix_size,opt.in_dim,))
    pad_img = Lambda(lambda x: tf.pad(x, [[0,0],[3,3],[3,3],[0,0]],'REFLECT'))(img)
    c1 = conv2d(pad_img, opt.g_fir_dim, 7, 1,padding='VALID',activation='relu')
    #c1 = InstanceNormalization()(c1)
    c2 = conv2d(c1, opt.g_fir_dim*2, 3, 2,activation='relu')
    #c2 = InstanceNormalization()(c2)
    c3 = conv2d(c2, opt.g_fir_dim*4, 3, 2,activation='relu')
    #c3 = InstanceNormalization()(c3)
    #residule bolck
    r1 = residule_block(c3, opt.g_fir_dim*4)
    r2 = residule_block(r1, opt.g_fir_dim*4)
    r3 = residule_block(r2, opt.g_fir_dim*4)
    r4 = residule_block(r3, opt.g_fir_dim*4)
    r5 = residule_block(r4, opt.g_fir_dim*4)
    r6 = residule_block(r5, opt.g_fir_dim*4)
    r7 = residule_block(r6, opt.g_fir_dim*4)
    r8 = residule_block(r7, opt.g_fir_dim*4)
    r9 = residule_block(r8, opt.g_fir_dim*4)

    t1 = trans_conv2d(r9, opt.g_fir_dim*2, 3, 2,padding='SAME',activation='relu')
    #t1 = InstanceNormalization()(t1)
    t2 = trans_conv2d(t1, opt.g_fir_dim, 3, 2,padding='SAME',activation='relu')
    #t2 = InstanceNormalization()(t2)
    t2_pad = Lambda(lambda x: tf.pad(x, [[0,0],[1,2],[1,2],[0,0]],'REFLECT'))(t2)
    gen_img = conv2d(t2_pad, opt.out_dim, 7, 1, padding='VALID',activation='tanh')
    return Model(inputs = img, outputs =gen_img)

def generator_unet():
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=(256,256,3))

        # Downsampling
        d1 = conv2d(d0, 32)
        d2 = conv2d(d1, 64)
        d3 = conv2d(d2, 128)
        d4 = conv2d(d3, 256)

        # Upsampling
        u1 = deconv2d(d4, d3, 128)
        u2 = deconv2d(u1, d2, 64)
        u3 = deconv2d(u2, d1, 32)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(3, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(d0, output_img)



def abs_criterion(x,y):
    return k.mean(k.abs(x-y))

def mae_criterion(x,y):
    return k.mean((x-y)**2)

def sce_criterion(logit, label):
    return k.mean(sigmoid(binary_crossentropy(label,logit)))