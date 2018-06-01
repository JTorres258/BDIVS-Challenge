#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ResNet models: version 1 and 2
ResNet v1
[a] Deep Residual Learning for Image Recognition
https://arxiv.org/pdf/1512.03385.pdf
ResNet v2
[b] Identity Mappings in Deep Residual Networks
https://arxiv.org/pdf/1603.05027.pdf
"""

from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten, Dropout
from keras.regularizers import l2
from keras.models import Model
from keras.initializers import RandomNormal
import numpy as np
import pdb

def lr_schedule(epoch, init_lr=1e-3):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
        init_lr (float32): initial learning rate
    # Returns
        lr (float32): learning rate
    """
    lr = init_lr
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            activation-bn-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, kernel_size, rp_filters, num_classes=1):
    """ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)    
    
    rand_proj = Conv2D(filters = rp_filters, kernel_size = kernel_size, padding='same', activation='linear', strides = kernel_size, 
                       trainable=False, kernel_initializer = RandomNormal(mean=0.0, stddev = np.sqrt(rp_filters), seed=5))(inputs=inputs)
    

#    inputs = Input(shape=input_shape)    
    
    x = resnet_layer(rand_proj)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    x = Flatten()(x)
    
    
    

    x1 = Dense(1024)(x)
    x1 = Activation('relu')(x1)
    x1 = Dropout(0.5)(x1)
    x1 = Dense(512)(x1)
    x1 = Activation('relu')(x1)
    x1 = Dropout(0.5)(x1)
    
    x1 = Dense(num_classes)(x1)
    x1 = Activation('sigmoid', name='a1')(x1)
    
    x2 = Dense(1024)(x)
    x2 = Activation('relu')(x2)
    x2 = Dropout(0.5)(x2)
    x2 = Dense(512)(x2)
    x2 = Activation('relu')(x2)
    x2 = Dropout(0.5)(x2)   
    
    x2 = Dense(num_classes)(x2)
    x2 = Activation('sigmoid', name='a2')(x2)
    
    x3 = Dense(1024)(x)
    x3 = Activation('relu')(x3)
    x3 = Dropout(0.5)(x3)
    x3 = Dense(512)(x3)
    x3 = Activation('relu')(x3)
    x3 = Dropout(0.5)(x3)
      
    x3 = Dense(num_classes)(x3)
    x3 = Activation('sigmoid', name='a3')(x3)
    
    x4 = Dense(1024)(x)
    x4 = Activation('relu')(x4)
    x4 = Dropout(0.5)(x4)
    x4 = Dense(512)(x4)
    x4 = Activation('relu')(x4)
    x4 = Dropout(0.5)(x4)
      
    x4 = Dense(num_classes)(x4)
    x4 = Activation('sigmoid', name='a4')(x4)
    
    x5 = Dense(1024)(x)
    x5 = Activation('relu')(x5)
    x5 = Dropout(0.5)(x5)
    x5 = Dense(512)(x5)
    x5 = Activation('relu')(x5)
    x5 = Dropout(0.5)(x5)
      
    x5 = Dense(num_classes)(x5)
    x5 = Activation('sigmoid', name='a5')(x5)
    
    x6 = Dense(1024)(x)
    x6= Activation('relu')(x6)
    x6 = Dropout(0.5)(x6)
    x6 = Dense(512)(x6)
    x6 = Activation('relu')(x6)
    x6 = Dropout(0.5)(x6)
      
    x6 = Dense(num_classes)(x6)
    x6= Activation('sigmoid', name='a6')(x6)
    
    x7 = Dense(1024)(x)
    x7 = Activation('relu')(x7)
    x7 = Dropout(0.5)(x7)
    x7 = Dense(512)(x7)
    x7 = Activation('relu')(x7)
    x7 = Dropout(0.5)(x7)
      
    x7 = Dense(num_classes)(x7)
    x7 = Activation('sigmoid', name='a7')(x7)
    
    x8 = Dense(1024)(x)
    x8 = Activation('relu')(x8)
    x8 = Dropout(0.5)(x8)
    x8 = Dense(512)(x8)
    x8 = Activation('relu')(x8)
    x8 = Dropout(0.5)(x8)
      
    x8 = Dense(num_classes)(x8)
    x8 = Activation('sigmoid', name='a8')(x8)
    
    x9 = Dense(1024)(x)
    x9 = Activation('relu')(x9)
    x9 = Dropout(0.5)(x9)
    x9 = Dense(512)(x9)
    x9 = Activation('relu')(x9)
    x9 = Dropout(0.5)(x9)
      
    x9 = Dense(num_classes)(x9)
    x9 = Activation('sigmoid', name='a9')(x9)
    
    x10 = Dense(1024)(x)
    x10 = Activation('relu')(x10)
    x10 = Dropout(0.5)(x10)
    x10 = Dense(512)(x10)
    x10 = Activation('relu')(x10)
    x10 = Dropout(0.5)(x10)
      
    x10 = Dense(num_classes)(x10)
    x10 = Activation('sigmoid', name='a10')(x10)
    
    x11 = Dense(1024)(x)
    x11 = Activation('relu')(x11)
    x11 = Dropout(0.5)(x11)
    x11 = Dense(512)(x11)
    x11 = Activation('relu')(x11)
    x11= Dropout(0.5)(x11)
      
    x11 = Dense(num_classes)(x11)
    x11 = Activation('sigmoid', name='a11')(x11)
    
    x12 = Dense(1024)(x)
    x12 = Activation('relu')(x12)
    x12 = Dropout(0.5)(x12)
    x12 = Dense(512)(x12)
    x12 = Activation('relu')(x12)
    x12 = Dropout(0.5)(x12)
      
    x12 = Dense(num_classes)(x12)
    x12 = Activation('sigmoid', name='a12')(x12)
    
    x13 = Dense(1024)(x)
    x13 = Activation('relu')(x13)
    x13 = Dropout(0.5)(x13)
    x13 = Dense(512)(x13)
    x13 = Activation('relu')(x13)
    x13 = Dropout(0.5)(x13)
      
    x13= Dense(num_classes)(x13)
    x13 = Activation('sigmoid', name='a13')(x13)
    
    x14 = Dense(1024)(x)
    x14 = Activation('relu')(x14)
    x14 = Dropout(0.5)(x14)
    x14 = Dense(512)(x14)
    x14 = Activation('relu')(x14)
    x14 = Dropout(0.5)(x14)
      
    x14 = Dense(num_classes)(x14)
    x14 = Activation('sigmoid', name='a14')(x14)
    
    x15 = Dense(1024)(x)
    x15 = Activation('relu')(x15)
    x15 = Dropout(0.5)(x15)
    x15 = Dense(512)(x15)
    x15 = Activation('relu')(x15)
    x15 = Dropout(0.5)(x15)
      
    x15 = Dense(num_classes)(x15)
    x15 = Activation('sigmoid', name='a15')(x15)
    
    x16 = Dense(1024)(x)
    x16 = Activation('relu')(x16)
    x16 = Dropout(0.5)(x16)
    x16 = Dense(512)(x16)
    x16 = Activation('relu')(x16)
    x16 = Dropout(0.5)(x16)
      
    x16 = Dense(num_classes)(x16)
    x16 = Activation('sigmoid', name='a16')(x16)
    
    x17 = Dense(1024)(x)
    x17 = Activation('relu')(x17)
    x17 = Dropout(0.5)(x17)
    x17 = Dense(512)(x17)
    x17 = Activation('relu')(x17)
    x17 = Dropout(0.5)(x17)
      
    x17 = Dense(num_classes)(x17)
    x17 = Activation('sigmoid', name='a17')(x17)
    
    x18 = Dense(1024)(x)
    x18 = Activation('relu')(x18)
    x18 = Dropout(0.5)(x18)
    x18 = Dense(512)(x18)
    x18 = Activation('relu')(x18)
    x18 = Dropout(0.5)(x18)
      
    x18 = Dense(num_classes)(x18)
    x18 = Activation('sigmoid', name='a18')(x18)
    
    x19 = Dense(1024)(x)
    x19 = Activation('relu')(x19)
    x19 = Dropout(0.5)(x19)
    x19 = Dense(512)(x19)
    x19 = Activation('relu')(x19)
    x19 = Dropout(0.5)(x19)
      
    x19 = Dense(num_classes)(x19)
    x19 = Activation('sigmoid', name='a19')(x19)
    
    x20 = Dense(1024)(x)
    x20 = Activation('relu')(x20)
    x20 = Dropout(0.5)(x20)
    x20 = Dense(512)(x20)
    x20 = Activation('relu')(x20)
    x20 = Dropout(0.5)(x20)
      
    x20 = Dense(num_classes)(x20)
    x20 = Activation('sigmoid', name='a20')(x20)
    
    x21 = Dense(1024)(x)
    x21 = Activation('relu')(x21)
    x21= Dropout(0.5)(x21)
    x21 = Dense(512)(x21)
    x21 = Activation('relu')(x21)
    x21 = Dropout(0.5)(x21)
      
    x21 = Dense(num_classes)(x21)
    x21 = Activation('sigmoid', name='a21')(x21)
    
    
    # Define steering-collision model
    model = Model(inputs=[inputs], outputs=[x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21])
    print(model.summary())
    return model


def resnet_v2(input_shape, depth, num_classes=1):
    """ResNet Version 2 Model builder [b]
    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model