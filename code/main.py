#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# main


from scipy.spatial.distance import cdist
from pycocotools.coco import maskUtils
from pycocotools.coco import COCO 
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import zipfile
import pandas
import json
import math
import cv2
import os
import re


# parameters

num_joints         = 18
num_joints_and_bkg = num_joints + 1
num_connections    = 19
idx_in_coco        = [0, lambda x: _get_neck(x, 5, 6), 6, 8,
                      10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
idx_in_coco_str    = [
                     'Nose','Neck','RShoulder','RElbow','RWrist','LShoulder','LElbow','LWrist',
                     'RHip','RKnee','RAnkle','LHip','LKnee','LAnkle','REye','LEye','REar','LEar']

joint_pairs        = list(zip(
                    [1, 8, 9, 1, 11, 12, 1, 2, 3, 2, 1, 5, 6, 5, 1, 0, 0, 14, 15],
                    [8, 9, 10, 11, 12, 13, 2, 3, 4, 16, 5, 6, 7, 17, 0, 14, 15, 16, 17]))



ALL_PAF_MASK       = np.repeat(
                         np.ones((46, 46, 1), dtype=np.uint8), 38, axis=2)

ALL_HEATMAP_MASK   = np.repeat(
                         np.ones((46, 46, 1), dtype=np.uint8), 19, axis=2)



# training



batch_size   = 10
base_lr      = 4e-5 # 2e-5
momentum     = 0.9
weight_decay = 5e-4
lr_policy    =  "step"
gamma        = 0.333
stepsize     = 136106 #68053   // after each stepsize iterations update learning rate: lr=lr*gamma
max_iter     = 200000 # 600000

weights_best_file = "/content/drive/My Drive/pose/weights_best.h5"
training_log      = "training.csv"
logs_dir          = "./logs"

from_vgg = {
    'conv1_1': 'block1_conv1',
    'conv1_2': 'block1_conv2',
    'conv2_1': 'block2_conv1',
    'conv2_2': 'block2_conv2',
    'conv3_1': 'block3_conv1',
    'conv3_2': 'block3_conv2',
    'conv3_3': 'block3_conv3',
    'conv3_4': 'block3_conv4',
    'conv4_1': 'block4_conv1',
    'conv4_2': 'block4_conv2'
}


def get_last_epoch():

  data = pandas.read_csv(training_log)
  return max(data['epoch'].values)


def restore_weights(weights_best_file, model):
    # load previous weights or vgg19 if this is the first run
  if os.path.exists(weights_best_file):
    print("Loading the best weights...")

    model.load_weights(weights_best_file)

  else:
    print("Loading vgg19 weights...")

    vgg_model = VGG19(include_top=False, weights='imagenet')

    for layer in model.layers:
      if layer.name in from_vgg:
        vgg_layer_name = from_vgg[layer.name]
        layer.set_weights(vgg_model.get_layer(vgg_layer_name).get_weights())
        print("Loaded VGG19 layer: " + vgg_layer_name)



def get_lr_multipliers(model):
  lr_mult = dict()
  for layer in model.layers:

    if isinstance(layer, Conv2D):

      # stage = 1
      if re.match("Mconv\d_stage1.*", layer.name):
        kernel_name          = layer.weights[0].name
        bias_name            = layer.weights[1].name
        lr_mult[kernel_name] = 1
        lr_mult[bias_name]   = 2

      # stage > 1
      elif re.match("Mconv\d_stage.*", layer.name):
        kernel_name          = layer.weights[0].name
        bias_name            = layer.weights[1].name
        lr_mult[kernel_name] = 4
        lr_mult[bias_name]   = 8

      # vgg
      else:
        kernel_name          = layer.weights[0].name
        bias_name            = layer.weights[1].name
        lr_mult[kernel_name] = 1
        lr_mult[bias_name]   = 2

  return lr_mult



def get_loss_funcs(nb_stages=6):
  def _eucl_loss(x, y):
    return K.sum(K.square(x - y)) / batch_size / 2

  keys = ["weight_stage{}_L{}".format(stage+1, L+1) for stage in range(nb_stages) for L in range(2)]

  losses = dict.fromkeys(keys, _eucl_loss)

  return losses


def step_decay(epoch, iterations_per_epoch):

  initial_lrate = base_lr
  steps         = epoch * iterations_per_epoch

  lrate         = initial_lrate * math.pow(gamma, math.floor(steps/stepsize))

  return lrate



def gen(df):
    """
    Wrapper around generator. Keras fit_generator requires looping generator.
    :param df: dataflow instance
    """
    while True:
        for i in df.get_data():
            yield i
            
            
            
            
def get_training_model(weight_decay):

    stages            = 6
    np_branch1        = 38
    np_branch2        = 19

    img_input_shape   = (None, None, 3)
    vec_input_shape   = (None, None, 38) # vector filed
    heat_input_shape  = (None, None, 19) # heat map

    inputs            = []
    outputs           = []

    img_input         = Input(shape=img_input_shape)
    vec_weight_input  = Input(shape=vec_input_shape)
    heat_weight_input = Input(shape=heat_input_shape)

    inputs.append(img_input)
    inputs.append(vec_weight_input)
    inputs.append(heat_weight_input)

    img_normalized    = Lambda(lambda x: x / 256 - 0.5)(img_input) # [-0.5, 0.5]

    # VGG
    stage0_out        = vgg_block(img_normalized, weight_decay)

    # stage 1 - branch 1 (PAF)
    stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1, weight_decay)
    w1                 = apply_mask(stage1_branch1_out, vec_weight_input, heat_weight_input, np_branch1, 1, 1)

    # stage 1 - branch 2 (confidence maps)
    stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, weight_decay)
    w2                 = apply_mask(stage1_branch2_out, vec_weight_input, heat_weight_input, np_branch2, 1, 2)

    x                  = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

    outputs.append(w1)
    outputs.append(w2)

    # stage sn >= 2
    for sn in range(2, stages + 1):
        # stage SN - branch 1 (PAF)
        stageT_branch1_out = stageT_block(x, np_branch1, sn, 1, weight_decay)
        w1                 = apply_mask(stageT_branch1_out, vec_weight_input, heat_weight_input, np_branch1, sn, 1)

        # stage SN - branch 2 (confidence maps)
        stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, weight_decay)
        w2                 = apply_mask(stageT_branch2_out, vec_weight_input, heat_weight_input, np_branch2, sn, 2)

        outputs.append(w1)
        outputs.append(w2)

        if (sn < stages):
            x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

    model = Model(inputs=inputs, outputs=outputs)

    return model


def get_testing_model():
    stages             = 6
    np_branch1         = 38
    np_branch2         = 19

    img_input_shape    = (None, None, 3)

    img_input          = Input(shape=img_input_shape)

    img_normalized     = Lambda(lambda x: x / 256 - 0.5)(img_input) # [-0.5, 0.5]

    # VGG
    stage0_out         = vgg_block(img_normalized, None)

    # stage 1 - branch 1 (PAF)
    stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1, None)

    # stage 1 - branch 2 (confidence maps)
    stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, None)

    x                  = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

    # stage t >= 2
    stageT_branch1_out = None
    stageT_branch2_out = None
    for sn in range(2, stages + 1):
        stageT_branch1_out = stageT_block(x, np_branch1, sn, 1, None)
        stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, None)

        if (sn < stages):
            x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

    model = Model(inputs=[img_input], outputs=[stageT_branch1_out, stageT_branch2_out])

    return model



try :
  all_meta__ = np.load("/content/drive/My Drive/all_meta/meta_data.npy" , allow_pickle=True)
except:
  all_meta__ = prepare(annotations,(368, 368))
  np.save("/content/drive/My Drive/all_meta/meta_data" , np.array(all_meta__))


image_paths = [meta.img_path for meta in all_meta__]
joints      = [meta.all_joints for meta in all_meta__]
mask_segma  = [meta.masks_segments for meta in all_meta__]
dims        = [(meta.height , meta.width) for meta in all_meta__]
dec__joints = dict(zip(image_paths ,joints ))
dec__segm   = dict(zip(image_paths ,mask_segma ))
dec__dims   = dict(zip(image_paths ,dims ))

len(image_paths)  , len(joints) , len(mask_segma) , len(dims)



# create_your_datasets
def read_preprocess_agument(image_path):
  image                                           = cv2.imread(str(image_path)[2:-1])
  joints_points                                   = joints_to_point8(dec__joints[str(image_path)[2:-1]])
  mask                                            = gen_mask__(dec__segm[str(image_path)[2:-1]] , dec__dims[str(image_path)[2:-1]])

  
  image , joints , mask                           = Rotation__(image , point8_to_joints(joints_points) , mask )
  image , joints , mask                           = scale__(image ,joints ,mask)
  image , joints , mask                           = flip__(image , joints , mask)
  image , joints , mask                           = resize__(image ,joints , mask )
  image                                           = apply_mask__(image , mask)

  image , mask_paf, mask_heatmap, pafmap, heatmap = build_sample_(image , mask , joints)
  
  return [image.astype(np.uint8) , mask_paf , mask_heatmap , pafmap ,heatmap]


data_set = tf.data.Dataset.from_tensor_slices(image_paths)
data_set = data_set.map(lambda x : tf.numpy_function(func = read_preprocess_agument , inp=[x],Tout=[tf.uint8,tf.uint8,tf.uint8,tf.double,tf.double]))




model = get_training_model(weight_decay)
#model.summary()
restore_weights(weights_best_file, model)

def decay_(base_lr):
  return tf.keras.optimizers.schedules.ExponentialDecay(base_lr,decay_steps=1000,decay_rate=0.96,staircase=True)

optimizer_1 = tf.keras.optimizers.SGD(learning_rate= decay_(base_lr) , momentum= momentum)
optimizer_2 = tf.keras.optimizers.SGD(learning_rate= decay_(base_lr * 2) , momentum= momentum)
optimizer_4 = tf.keras.optimizers.SGD(learning_rate= decay_(base_lr * 4) , momentum= momentum)
optimizer_8 = tf.keras.optimizers.SGD(learning_rate= decay_(base_lr * 8) , momentum= momentum)
 
    

    

for epoch in range(10):
  batch_numer = 0
  loss__ = 0
  for image, mask_paf ,mask_heatmap, pafmap ,  heatmaps in data_set.batch(batch_size):
    batch__ = [image , mask_paf , mask_heatmap]
    loss = 0
    with tf.GradientTape() as tape:
      out = model(batch__)
      for i in range(6):
        loss  += K.sum(K.square(out[i*2] - tf.cast(pafmap , dtype= tf.float32))) / batch_size/ 2
        loss  += K.sum(K.square(out[(i*2)+1] - tf.cast(heatmaps , dtype= tf.float32))) /batch_size/ 2
    
    variables_opt_1   = []
    variables_opt_2   = []
    variables_opt_4   = []
    variables_opt_8   = []
    for layer in model.layers:
      if isinstance(layer, Conv2D):
        # stage = 1
        if re.match("Mconv\d_stage1.*", layer.name):
          variables_opt_1.append(layer.weights[0].name)
          variables_opt_2.append(layer.weights[1].name)

        # stage > 1
        elif re.match("Mconv\d_stage.*", layer.name):
          variables_opt_4.append(layer.weights[0].name)
          variables_opt_8.append(layer.weights[1].name)

        # vgg
        else:
          variables_opt_1.append(layer.weights[0].name)
          variables_opt_2.append(layer.weights[1].name)
      #print(variables_opt_1)

    #trainable_variables = model.trainable_variables
    gradients             = tape.gradient(loss , model.trainable_variables )
    
    gradients_1           = [gradients[i]  for i in range(len(gradients))  if model.trainable_variables[i].name in variables_opt_1 ]
    variables_opt_1       = [model.trainable_variables[i] for i in range(len(model.trainable_variables)) if model.trainable_variables[i].name in variables_opt_1]

    gradients_2           = [gradients[i]  for i in range(len(gradients))  if model.trainable_variables[i].name in variables_opt_2 ]
    variables_opt_2       = [model.trainable_variables[i] for i in range(len(model.trainable_variables)) if model.trainable_variables[i].name in variables_opt_2]
    

    gradients_4           = [gradients[i]  for i in range(len(gradients))  if model.trainable_variables[i].name in variables_opt_4 ]
    variables_opt_4       = [model.trainable_variables[i] for i in range(len(model.trainable_variables)) if model.trainable_variables[i].name in variables_opt_4]
    

    gradients_8           = [gradients[i]  for i in range(len(gradients))  if model.trainable_variables[i].name in variables_opt_8 ]
    variables_opt_8       = [model.trainable_variables[i] for i in range(len(model.trainable_variables)) if model.trainable_variables[i].name in variables_opt_8]
    
    
    optimizer_1.apply_gradients(zip(gradients_1 , variables_opt_1))
    optimizer_2.apply_gradients(zip(gradients_2 , variables_opt_2))
    optimizer_4.apply_gradients(zip(gradients_4 , variables_opt_4))
    optimizer_8.apply_gradients(zip(gradients_8 , variables_opt_8))
    batch_numer  +=1
    loss__       += loss.numpy()

    if (batch_numer +1 ) % 100 == 0:
      print("epoch : %d batch : %d  loss : %f" %(epoch , batch_numer +1,  loss__ / batch_numer ))

    if (batch_numer +1 ) % 200 == 0:
      model.save_weights(weights_best_file)
      print("model saved")


  print("epoch %d  loss : %f" %(epoch , loss__ /batch_numer ))



