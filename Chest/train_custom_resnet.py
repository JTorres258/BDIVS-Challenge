import tensorflow as tf
import numpy as np
import os
import sys
import gflags

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from keras import backend as K
import keras

import logz
import nets
import resnet_models
import utils
import data_utils
import log_utils
from common_flags import FLAGS
from time import time, strftime, localtime
import pdb

# Constants
TRAIN_PHASE = 1


def getModel(img_width, img_height, img_channels, output_dim, weights_path):
    """
    Initialize model.

    # Arguments
       img_width: Target image widht.
       img_height: Target image height.
       img_channels: Target image channels.
       output_dim: Dimension of model output (number of classes).
       weights_path: Path to pre-trained model.

    # Returns
       model: A Model instance.
    """
    model = nets.resnet50(img_width, img_height, img_channels, output_dim)

    if weights_path:
        try:
            model.load_weights(weights_path)
            print("Loaded model from {}".format(weights_path))
        except:
            print("Impossible to find weight path. Returning untrained model")

    return model


def getModelResnet(n, version, img_width, img_height, img_channels, output_dim, weights_path):
    """
    Initialize model.

    # Arguments
       n: parameter that determines the net depth.
       version: 1 for resnet v1 or 2 for v2.
       img_width: Target image widht.
       img_height: Target image height.
       img_channels: Target image channels.
       output_dim: Dimension of model output (number of classes).
       weights_path: Path to pre-trained model.

    # Returns
       model: A Model instance.
    """
    
    # Computed depth from supplied model parameter n
    if version == 1:
        depth = n * 6 + 2
    elif version == 2:
        depth = n * 9 + 2
    
    input_shape = (img_height, img_width, img_channels);
    
    kernel_size = (FLAGS.kernel_height, FLAGS.kernel_width)
    RP_filters = FLAGS.RP_filters

    if version == 2:
        model = resnet_models.resnet_v2(input_shape=input_shape, depth=depth, num_classes=output_dim)
    else:
        model = resnet_models.resnet_v1(input_shape=input_shape, depth=depth, num_classes=output_dim,
                                        kernel_size = kernel_size, rp_filters = RP_filters)

    # Model name, depth and version
    model_type = 'ResNet%dv%d' % (depth, version)
    print(model_type)

    if weights_path:
        try:
            model.load_weights(weights_path)
            print("Loaded model from {}".format(weights_path))
        except:
            print("Impossible to find weight path. Returning untrained model")

    return model


def trainModel(train_data_generator, val_data_generator, model, initial_epoch):
    """
    Model training.

    # Arguments
       train_data_generator: Training data generated batch by batch.
       val_data_generator: Validation data generated batch by batch.
       model: A Model instance.
       initial_epoch: Epoch from which training starts.
    """
    # Configure training process
    model.compile(loss='binary_crossentropy',
          optimizer=Adam(lr=resnet_models.lr_schedule(0, FLAGS.initial_lr)),
          metrics=['binary_accuracy'],
          loss_weights=np.ones((15,)).tolist())

    # Save model with the lowest validation loss
    weights_path = os.path.join(FLAGS.experiment_rootdir, 'weights_{epoch:03d}.h5')
    writeBestModel = ModelCheckpoint(filepath=weights_path, monitor='val_loss',
                                     save_best_only=True, save_weights_only=True)
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    # Save training and validation losses.
    logz.configure_output_dir(FLAGS.experiment_rootdir)
    saveModelAndLoss = log_utils.MyCallback(filepath=FLAGS.experiment_rootdir)

    # Train model
    steps_per_epoch = int(np.ceil(train_data_generator.samples / FLAGS.batch_size))
    validation_steps = int(np.ceil(val_data_generator.samples / FLAGS.batch_size))-1
    
    lr_scheduler = LearningRateScheduler(resnet_models.lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)
    strTime = strftime("%Y%b%d_%Hh%Mm%Ss", localtime(time()))
#    tensorboard = TensorBoard(log_dir="logs/{}".format(strTime), histogram_freq=10, 
#                              batch_size=32, write_graph=False, write_grads=True, 
#                              write_images=False, embeddings_freq=0,
#                              embeddings_layer_names=None, embeddings_metadata=None)
    
    #tensorboard = TensorBoard(log_dir="logs/{}".format(strTime), histogram_freq=10)
    tensorboard = TensorBoard(log_dir="logs/{}".format(strTime), histogram_freq=0)
    callbacks = [writeBestModel, saveModelAndLoss, lr_reducer, lr_scheduler, tensorboard]
    model.fit_generator(train_data_generator,
                        epochs=FLAGS.epochs, steps_per_epoch = steps_per_epoch,
                        callbacks=callbacks,
                        validation_data=val_data_generator,
                        validation_steps = validation_steps,
                        initial_epoch=initial_epoch)
    
    
def _main():
    
    # Set random seed
    if FLAGS.random_seed:
        seed = np.random.randint(0,2*31-1)
    else:
        seed = 5
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Set training phase
    K.set_learning_phase(TRAIN_PHASE)

    # Create the experiment rootdir if not already there
    if not os.path.exists(FLAGS.experiment_rootdir):
        os.makedirs(FLAGS.experiment_rootdir)

    # Input image dimensions
    img_width, img_height = FLAGS.img_width, FLAGS.img_height

    
    # Image mode (RGB or grayscale)
    if FLAGS.img_mode=='rgb':
        img_channels = 3
    elif FLAGS.img_mode == 'grayscale':
        img_channels = 1
    else:
        raise IOError("Unidentified image mode: use 'grayscale' or 'rgb'")

    # Output dimension (empty place probability)
    output_dim=1

    # Generate training data with real-time augmentation
    train_datagen = data_utils.DataGenerator(rescale = 1./255)
    
    # Iterator object containing training data to be generated batch by batch
    train_generator = train_datagen.flow_from_directory(FLAGS.train_dir,
                                                        output_dim,
                                                        shuffle = True,
                                                        img_mode = FLAGS.img_mode,
                                                        target_size=(img_height, img_width),
                                                        batch_size = FLAGS.batch_size)
    
    # Check if the number of classes in dataset corresponds to the one specified                                                    
#    assert train_generator.num_classes == num_classes, \
#                        " Not macthing output dimensions in training data."                                                    


    # Generate validation data with real-time augmentation
    val_datagen = data_utils.DataGenerator(rescale = 1./255)
    
    # Iterator object containing validation data to be generated batch by batch
    val_generator = val_datagen.flow_from_directory(FLAGS.val_dir,
                                                    output_dim,
                                                    shuffle = False,
                                                    img_mode = FLAGS.img_mode,
                                                    target_size=(img_height, img_width),
                                                    batch_size = FLAGS.batch_size)

    # Check if the number of classes in dataset corresponds to the one specified
#    assert val_generator.num_classes == num_classes, \
#                        " Not macthing output dimensions in validation data."
                        

    # Weights to restore
    weights_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.weights_fname)
    
    # Epoch from which training starts
    initial_epoch = 0
    if not FLAGS.restore_model:
        # In this case weights are initialized randomly
        weights_path = None
    else:
        # In this case weigths are initialized as specified in pre-trained model
        initial_epoch = FLAGS.initial_epoch

    # Define model
    model = getModelResnet(FLAGS.n_layers, FLAGS.rn_version, img_width, img_height, img_channels,
                        output_dim, weights_path)  

    # Serialize model into json
    json_model_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.json_model_fname)
    utils.modelToJson(model, json_model_path)

    # Train model
    trainModel(train_generator, val_generator, model, initial_epoch)
    
    # Plot training and validation losses
    utils.plot_loss(FLAGS.experiment_rootdir)


def main(argv):
    # Utility main to load flags
    try:
      argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError:
      print ('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))

      sys.exit(1)
    _main()


if __name__ == "__main__":
    main(sys.argv)
