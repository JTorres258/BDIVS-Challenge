import os
import numpy as np
import cv2
import pdb

import keras
from keras import backend as K
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import ImageDataGenerator


class DataGenerator(ImageDataGenerator):
    """
    Generate minibatches of images and labels with real-time augmentation.

    The only function that changes w.r.t. parent class is the flow that
    generates data. This function needed in fact adaptation for different
    directory structure and labels. All the remaining functions remain
    unchanged.
    """
    def flow_from_directory(self, directory, output_dim, target_size=(224,224),
                            img_mode='grayscale', batch_size=32, shuffle=True,
                            seed=None, follow_links=False):
        return DirectoryIterator(
                directory, output_dim, self, target_size=target_size, img_mode=img_mode,
                batch_size=batch_size, shuffle=shuffle, seed=seed,
                follow_links=follow_links)



class DirectoryIterator(Iterator):
    """
    Class for managing data loading of images and labels
    We assume that the folder structure is:
    root_folder/
           user_1/
               class_1/
                   frame_00000000.png
                   frame_00000001.png
                   .
                   .
                   frame_00999999.png
               class_2/
               .
               .
               class_n/
                    
           user_2/
           .
           .
           user_n/


    # Arguments
       directory: Path to the root directory to read data from.
       num_classes: Output dimension (number of classes).
       image_data_generator: Image Generator.
       target_size: tuple of integers, dimensions to resize input images to.
       img_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
       batch_size: The desired batch size
       shuffle: Whether to shuffle data or not
       seed : numpy seed to shuffle data
       follow_links: Bool, whether to follow symbolic links or not

    # TODO: Add functionality to save images to have a look at the augmentation
    """
    def __init__(self, directory, output_dim, image_data_generator,
            target_size=(224,224), img_mode = 'grayscale',
            batch_size=32, shuffle=True, seed=None, follow_links=False):
        self.directory = os.path.realpath(directory)
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.follow_links = follow_links
        
        # Initialize image mode
        if img_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', img_mode,
                             '; expected "rgb" or "grayscale".')
        self.img_mode = img_mode
        if self.img_mode == 'rgb':
            self.image_shape = self.target_size + (3,)
        else:
            self.image_shape = self.target_size + (1,)
        
        # Initialize number of classes
        self.output_dim = output_dim
        
        # Allowed image formats
        self.formats = {'png', 'jpg'}

        # Number of samples in dataset
        self.samples = 0

        # First count how many users there are
        experiments = []
        for subdir in sorted(os.listdir(directory)):
            if os.path.isdir(os.path.join(directory, subdir)):
                experiments.append(subdir)
        self.num_experiments = len(experiments)
        
        # Filenames of all samples/images in dataset. 
        self.filenames = []     
        # Labels (ground truth) of all samples/images in dataset
        self.ground_truth = []
        
        # Decode dataset structure.
        # All the filenames and ground truths are loaded in memory from the
        # begining.
        # Images instead, will be loaded iteratively as the same time the
        # training process needs a new batch.
#        pdb.set_trace()
        
        for experiment in experiments:
            experiment_path = os.path.join(directory, experiment)
           
            if os.path.isdir(experiment_path):
                try:
                    # Read and count all filenames in dataset
                    self._decode_experiment_dir(experiment_path)
                except:
                    continue                          
                
#        try:   
#            # Read and count all filenames in dataset
#            self._decode_experiment_dir(directory)
#        except:
#            pass
  
        # Check if dataset is empty            
        if self.samples == 0:
            raise IOError("Did not find any data")
        
        # Conversion of list into array
        self.ground_truth = np.array(self.ground_truth, dtype= K.floatx())
         

        print('Found {} images belonging to {} experiments'.format(
                self.samples, self.num_experiments))

        super(DirectoryIterator, self).__init__(self.samples,
                batch_size, shuffle, seed)


    def _recursive_list(self, subpath):
        return sorted(os.walk(subpath, followlinks=self.follow_links),
                key=lambda tpl: tpl[0])
                
                
    def _decode_experiment_dir(self, experiment_path):
        """
        Extract valid filenames in every class.
        
        # Arguments
            image_dir_path: path to class folder to be decoded
        """
       
        gt_dir_file=os.path.join(experiment_path,'groundtruth.txt')
        try: 
            #read groundTruth file
           gt= np.loadtxt(gt_dir_file, delimiter=" ", usecols= (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15))
           gt_fname = np.genfromtxt(gt_dir_file, delimiter=" ", usecols= (0), dtype='str')

        except:
                raise IOError("Couldn't read groundtruth file")
                
        image_dir_path=os.path.join(experiment_path,'images')
        for root, _, files in self._recursive_list(image_dir_path):            
            for frame_number, fname in enumerate(files):
                is_valid = False
                for extension in self.formats:
                    if fname.lower().endswith('.' + extension):
                        is_valid = True
                        break
                if is_valid:
#                    absolute_path = os.path.join(root, fname)
                    absolute_path = os.path.join(root, gt_fname[frame_number])
                    self.filenames.append(os.path.relpath(absolute_path,
                                                          self.directory))
                    self.ground_truth.append(gt[frame_number,:])
#                    pdb.set_trace()
                    self.samples += 1
                    

    def _get_batches_of_transformed_samples(self, index_array):
        """
        Public function to fetch next batch.
        
        Image transformation is not under thread lock, so it can be done in
        parallel
        
        # Returns
            The next batch of images and categorical labels.
        """
        
        current_batch_size = index_array.shape[0]
            
        # Initialize batch of images
        batch_x = np.zeros((current_batch_size,) + self.image_shape,
                dtype=K.floatx())
        # Initialize batch of ground truth
        batch_y = []
        batch_region = np.zeros((current_batch_size,), dtype=K.floatx())
                                 
        grayscale = self.img_mode == 'grayscale'

        for k in range(21):
            # Build batch of image data
            for i, j in enumerate(index_array):
                fname = self.filenames[j]
                x = load_img(os.path.join(self.directory, fname),
                                       grayscale=grayscale,
                                       target_size=self.target_size)
                # Data augmentation
 #               x = self.image_data_generator.random_transform(x)
                x = self.image_data_generator.standardize(x)
                batch_x[i] = x

                # Build batch of labels
                batch_region[i]=self.ground_truth[j, k]
                
            batch_y.append(batch_region)
        
        return batch_x, batch_y
    
    def next(self):
        """
        Public function to fetch next batch
        # Returns
            The next batch of images and commands.
        """
        
        with self.lock:
            index_array = next(self.index_generator)

        return self._get_batches_of_transformed_samples(index_array)


def load_img(path, grayscale=False, target_size=None):
    """
    Load an image.

    # Arguments
        path: Path to image file.
        grayscale: Boolean, wether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.

    # Returns
        Image as numpy array.
    """
    
    # Read input image
    img = cv2.imread(path)
    
    if grayscale:
        if len(img.shape) != 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if target_size:
        if (img.shape[0], img.shape[1]) != target_size:
            img = cv2.resize(img, (target_size[1], target_size[0]))

    if grayscale:
        img = img.reshape((img.shape[0], img.shape[1], 1))

    return np.asarray(img, dtype=np.float32)
