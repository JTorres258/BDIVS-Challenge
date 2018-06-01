import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.applications.resnet50 import ResNet50
from keras import regularizers
from keras.layers.merge import add
import pdb

def resnet8(img_width, img_height, img_channels, output_dim):
    """
    Define model architecture.
    
    # Arguments
       img_width: Target image widht.
       img_height: Target image height.
       img_channels: Target image channels.
       output_dim: Dimension of model output.
       
    # Returns
       model: A Model instance.
    """

    # Input
    img_input = Input(shape=(img_height, img_width, img_channels))

    x1 = Conv2D(32, (5, 5), strides=[2,2], padding='same')(img_input)
    x1 = MaxPooling2D(pool_size=(3, 3), strides=[2,2])(x1)

    # First residual block
    x2 = keras.layers.normalization.BatchNormalization()(x1)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x2)

    x2 = keras.layers.normalization.BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x2)

    x1 = Conv2D(32, (1, 1), strides=[2,2], padding='same')(x1)
    x3 = add([x1, x2])

    # Second residual block
    x4 = keras.layers.normalization.BatchNormalization()(x3)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x4)

    x4 = keras.layers.normalization.BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x4)

    x3 = Conv2D(64, (1, 1), strides=[2,2], padding='same')(x3)
    x5 = add([x3, x4])

    # Third residual block
    x6 = keras.layers.normalization.BatchNormalization()(x5)
    x6 = Activation('relu')(x6)
    x6 = Conv2D(128, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x6)

    x6 = keras.layers.normalization.BatchNormalization()(x6)
    x6 = Activation('relu')(x6)
    x6 = Conv2D(128, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x6)

    x5 = Conv2D(128, (1, 1), strides=[2,2], padding='same')(x5)
    x7 = add([x5, x6])

    x = Flatten()(x7)
    x = Dense(1024)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    

    x1 = Dense(output_dim)(x)
    x1 = Activation('sigmoid', name='a1')(x1)
    
    x2 = Dense(output_dim)(x)
    x2 = Activation('sigmoid', name='a2')(x2)
      
    x3 = Dense(output_dim)(x)
    x3 = Activation('sigmoid', name='a3')(x3)
      
    x4 = Dense(output_dim)(x)
    x4 = Activation('sigmoid', name='a4')(x4)
      
    x5 = Dense(output_dim)(x)
    x5 = Activation('sigmoid', name='a5')(x5)
      
    x6 = Dense(output_dim)(x)
    x6= Activation('sigmoid', name='a6')(x6)
      
    x7 = Dense(output_dim)(x)
    x7 = Activation('sigmoid', name='a7')(x7)
      
    x8 = Dense(output_dim)(x)
    x8 = Activation('sigmoid', name='a8')(x8)
      
    x9 = Dense(output_dim)(x)
    x9 = Activation('sigmoid', name='a9')(x9)
      
    x10 = Dense(output_dim)(x)
    x10 = Activation('sigmoid', name='a10')(x10)
      
    x11 = Dense(output_dim)(x)
    x11 = Activation('sigmoid', name='a11')(x11)
      
    x12 = Dense(output_dim)(x)
    x12 = Activation('sigmoid', name='a12')(x12)
      
    x13= Dense(output_dim)(x)
    x13 = Activation('sigmoid', name='a13')(x13)
      
    x14 = Dense(output_dim)(x)
    x14 = Activation('sigmoid', name='a14')(x14)
      
    x15 = Dense(output_dim)(x)
    x15 = Activation('sigmoid', name='a15')(x15)
      
    x16 = Dense(output_dim)(x)
    x16 = Activation('sigmoid', name='a16')(x16)
      
    x17 = Dense(output_dim)(x)
    x17 = Activation('sigmoid', name='a17')(x17)
      
    x18 = Dense(output_dim)(x)
    x18 = Activation('sigmoid', name='a18')(x18)
      
    x19 = Dense(output_dim)(x)
    x19 = Activation('sigmoid', name='a19')(x19)
      
    x20 = Dense(output_dim)(x)
    x20 = Activation('sigmoid', name='a20')(x20)
      
    x21 = Dense(output_dim)(x)
    x21 = Activation('sigmoid', name='a21')(x21)
    
    
    # Define steering-collision model
    model = Model(inputs=[img_input], outputs=[x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21])
    print(model.summary())

    return model


def gesture_net(img_width, img_height, img_channels, output_dim):
    """
    Define model architecture.

    # Arguments
       img_width: Target image widht.
       img_height: Target image height.
       img_channels: Target image channels.
       output_dim: Dimension of model output.

    # Returns
       model: A Model instance.
    """

    # Input
    img_input = Input(shape=(img_height, img_width, img_channels))
    
    x = Conv2D(32, (5, 5), strides=[3,3], padding='valid')(img_input)
    x = Activation('relu')(x)
    x = Dropout(0.20)(x)
    x = Conv2D(32, (3, 3), strides=[2,2], padding='valid')(x)
    x = Activation('relu')(x) 
    x = MaxPooling2D(pool_size=(2, 2), strides=[2,2])(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(64, (3, 3), strides=[1,1], padding='valid')(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64, (3, 3), strides=[1,1], padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=[2,2])(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(128, (5, 5), strides=[1,1], padding='same')(x)
    x = Activation('relu')(x)
    x = Dropout(0.35)(x)
    x = Conv2D(128, (3, 3), strides=[1,1], padding='same')(x)
    x = Activation('relu')(x)
    x = Dropout(0.30)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=[2,2])(x)
    
    x = Conv2D(256, (3, 3), strides=[1,1], padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), strides=[1,1], padding='same')(x)
    x = Activation('relu')(x)
    x = Dropout(0.30)(x)
    x = Conv2D(256, (3, 3), strides=[1,1], padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=[2,2])(x)
    x = Dropout(0.5)(x)
    
    x = Flatten()(x)
    x = Dense(128)(x)
    x = Dropout(0.5)(x)
    x = Activation('relu')(x)
    
    x = Dense(64)(x)
    x = Dropout(0.5)(x)
    x = Activation('relu')(x)
    x = Dense(output_dim)(x)
    x = Activation('softmax')(x)

    # Define steering-collision model
    model = Model(inputs=[img_input], outputs=[x])
    print(model.summary())

    return model
    

    
def resnet50(img_width, img_height, img_channels, output_dim):
    """
    Define model architecture.

    # Arguments
       img_width: Target image widht.
       img_height: Target image height.
       img_channels: Target image channels.
       output_dim: Dimension of model output.

    # Returns
       model: A Model instance.
    """
    
    # Input
    img_input = Input(shape=(img_height, img_width, img_channels))
    
    # ResNet50
    model = ResNet50(include_top=False, weights='imagenet', input_tensor=img_input)
    x = model.output
    
    # FC layers
    x = Flatten()(x)
    x = Dense(1024)(x)
    x = Activation('sigmoid')(x)
    x = Dropout(0.5)(x)
    
    x1 = Dense(output_dim)(x)
    x1 = Activation('sigmoid', name='a1')(x1)
    
    x2 = Dense(output_dim)(x)
    x2 = Activation('sigmoid', name='a2')(x2)
      
    x3 = Dense(output_dim)(x)
    x3 = Activation('sigmoid', name='a3')(x3)
      
    x4 = Dense(output_dim)(x)
    x4 = Activation('sigmoid', name='a4')(x4)
      
    x5 = Dense(output_dim)(x)
    x5 = Activation('sigmoid', name='a5')(x5)
      
    x6 = Dense(output_dim)(x)
    x6= Activation('sigmoid', name='a6')(x6)
      
    x7 = Dense(output_dim)(x)
    x7 = Activation('sigmoid', name='a7')(x7)
      
    x8 = Dense(output_dim)(x)
    x8 = Activation('sigmoid', name='a8')(x8)
      
    x9 = Dense(output_dim)(x)
    x9 = Activation('sigmoid', name='a9')(x9)
      
    x10 = Dense(output_dim)(x)
    x10 = Activation('sigmoid', name='a10')(x10)
      
    x11 = Dense(output_dim)(x)
    x11 = Activation('sigmoid', name='a11')(x11)
      
    x12 = Dense(output_dim)(x)
    x12 = Activation('sigmoid', name='a12')(x12)
      
    x13= Dense(output_dim)(x)
    x13 = Activation('sigmoid', name='a13')(x13)
      
    x14 = Dense(output_dim)(x)
    x14 = Activation('sigmoid', name='a14')(x14)
      
    x15 = Dense(output_dim)(x)
    x15 = Activation('sigmoid', name='a15')(x15)
      
    x16 = Dense(output_dim)(x)
    x16 = Activation('sigmoid', name='a16')(x16)
      
    x17 = Dense(output_dim)(x)
    x17 = Activation('sigmoid', name='a17')(x17)
      
    x18 = Dense(output_dim)(x)
    x18 = Activation('sigmoid', name='a18')(x18)
      
    x19 = Dense(output_dim)(x)
    x19 = Activation('sigmoid', name='a19')(x19)
      
    x20 = Dense(output_dim)(x)
    x20 = Activation('sigmoid', name='a20')(x20)
      
    x21 = Dense(output_dim)(x)
    x21 = Activation('sigmoid', name='a21')(x21)
    
    
    # Define steering-collision model
    model = Model(inputs=[img_input], outputs=[x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21])
    print(model.summary())

    return model
    


