import numpy as np
import cv2
import gflags
import utils
import sys
import os
import glob

import keras.backend as K

from common_flags import FLAGS

import pdb

def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer
    
    

def visualize_class_attention_map(img_path, model, out_path):
    target_size = (FLAGS.img_width, FLAGS.img_height)
    grayscale = FLAGS.img_mode == "grayscale"

    img = cv2.imread(img_path)
    #print(img_path)
    if target_size:
        if (img.shape[0], img.shape[1]) != target_size:
            img = cv2.resize(img, target_size)

    new_img = img.copy()
    if grayscale:
        if len(new_img.shape) != 2:
            new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
            new_img = new_img.reshape((new_img.shape[0],
                                     new_img.shape[1], 1))

    new_img = np.asarray(new_img / 255.0, dtype = np.float32)
    new_img  = np.expand_dims(new_img,axis=0)    
    
    class_weights = model.layers[-2].get_weights()[0]
    dense_weights = model.layers[-5].get_weights()[0]
    
    class_to_conv_weights = np.dot(dense_weights,class_weights)
    
    final_conv_layer = get_output_layer(model, "activation_49")
    get_output = K.function([model.layers[0].input],
                            [final_conv_layer.output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output([new_img])
    conv_outputs = conv_outputs[0, :, :, :]
    
    target_class = np.argmax(predictions)
    #print(target_class)
    
    # Create the class attention map
    cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[0:2])
    for i, w in enumerate(class_to_conv_weights[:, target_class]):
        cam += w*conv_outputs[:,:,i]
        
        
    #print("predictions", predictions)
    cam /= np.max(cam)
    cam = cv2.resize(cam, (FLAGS.img_width, FLAGS.img_height), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam < 0.2)] = 0
    img = heatmap*0.5 + img
    img = cv2.resize(img,(384,216))
    cv2.imwrite(out_path, img)
    del img, new_img, class_to_conv_weights, class_weights, dense_weights, final_conv_layer, conv_outputs, predictions



def _main():
    
    K.set_learning_phase(0)

#    input_dict = {"fist" : "000000000.jpg",
#                  "l" : "000000003.jpg",
#                  "ok" : "000000001.jpg",
#                  "palm" : "000000001.jpg",
#                  "pointer" : "000000001.jpg",
#                  "thumb down" : "000000000.jpg",
#                  "thumb up" : "000000003.jpg"}

    # Load json and create model
    json_model_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.json_model_fname)
    model = utils.jsonToModel(json_model_path)
    
    # Load weights
    weights_load_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.weights_fname)
    try:
        model.load_weights(weights_load_path)
        print("Loaded model from {}".format(weights_load_path))
    except:
        print("Impossible to find weight path. Returning untrained model")
        
    #print(model.summary())

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    experiments = glob.glob(FLAGS.test_dir + '/*')
    for exp in experiments:
        img_path = os.path.join(exp, 'images')
        images = glob.glob(img_path + '/*')
            
        for i, input_path in enumerate(sorted(images)):
            img_name = '{0:09d}.jpg'.format(i)
            output_path = os.path.join(FLAGS.experiment_rootdir, 'activations')
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            img_out_name = os.path.join(output_path, img_name)
            visualize_class_attention_map(input_path, model, img_out_name)
            
            
#    for key, value in input_dict.items():
#        img_input_path = os.path.join(FLAGS.test_dir, 'pablo_2', key, value)
#        img_out_path = os.path.join(FLAGS.experiment_rootdir, 'activations',
#                                    "{}.png".format(key))
#        visualize_class_attention_map(img_input_path, model, img_out_path)
        

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