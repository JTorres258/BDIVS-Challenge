import gflags


FLAGS = gflags.FLAGS

# Random seed
gflags.DEFINE_bool('random_seed', True, 'Random seed')

# Input
gflags.DEFINE_integer('img_width', 256, 'Target Image Width')
gflags.DEFINE_integer('img_height', 256, 'Target Image Height')
gflags.DEFINE_string('img_mode', "rgb", 'Load mode for images, either '
                     'rgb or grayscale')

gflags.DEFINE_integer('kernel_width', 24, 'Kernel Width in Random Projections')
gflags.DEFINE_integer('kernel_height', 18, 'Kernel Height in Random Projections')
gflags.DEFINE_integer('RP_filters', 70, 'Number of Filters in Random Projections')

# Training parameters
gflags.DEFINE_integer('batch_size', 1, 'Batch size in training and evaluation')
gflags.DEFINE_integer('epochs', 35, 'Number of epochs for training')
gflags.DEFINE_integer('initial_epoch', 0, 'Initial epoch to start training')
gflags.DEFINE_float('initial_lr', 1e-3, 'Initial learning rate for adam')

# Files
gflags.DEFINE_string('experiment_rootdir', ".\\models/test_1", 'Folder '
                     ' containing all the logs, model weights and results')
gflags.DEFINE_string('train_dir', ".\\training", 'Folder containing'
                     ' training experiments')
gflags.DEFINE_string('val_dir', ".\\validation", 'Folder containing'
                     ' validation experiments')
gflags.DEFINE_string('test_dir', ".\\testing", 'Folder containing'
                     ' testing experiments')
gflags.DEFINE_string('video_dir', ".\\video_1", 'Folder containing'
                     ' only one experiment to be processed')
gflags.DEFINE_string('exp_name', "exp_1", 'Name of the experiment'
                     ' to be processed')

# Model
gflags.DEFINE_bool('restore_model', False, 'Whether to restore a trained'
                   ' model for training')
gflags.DEFINE_string('weights_fname', "model_weights.h5", '(Relative) '
                                          'filename of model weights')
gflags.DEFINE_string('json_model_fname', "model_struct.json",
                          'Model struct json serialization, filename')
gflags.DEFINE_integer('n_layers', 1, 'Number that controls the depth network')
gflags.DEFINE_integer('rn_version', 1, 'Resnet architecture version: 1 or 2')

