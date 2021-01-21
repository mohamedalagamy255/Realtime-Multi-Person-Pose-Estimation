#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# build you model

from tensorflow.keras.layers import Activation ,Concatenate , Input , Lambda ,Conv2D ,MaxPooling2D ,Multiply 
from tensorflow.keras.initializers import random_normal , constant
from tensorflow.keras.applications import VGG19
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


def conv(x , nf , ks , name , weight_decay):
    kernel_reg = l2(weight_decay[0]) if weight_decay else None
    bias_reg   = l2(weight_decay[1]) if weight_decay else None
    
    x          = Conv2D(nf , (ks , ks) ,padding = "same" , name = name 
                  ,kernel_regularizer = kernel_reg 
                  , bias_regularizer = bias_reg 
                  ,kernel_initializer = random_normal(stddev=0.01)
                  ,bias_initializer = constant(0.0))(x)
    return x


def pooling(x , ks ,st , name):
    x = MaxPooling2D((ks,ks) , strides = (st , st) , name = name)(x)
    return x

def relu(x):
    return Activation("relu")(x)

def apply_mask(x, mask1, mask2, num_p, stage, branch):
    w_name = "weight_stage%d_L%d" % (stage, branch)
    if num_p == 38:
        w = Multiply(name=w_name)([x, mask1]) # vec_weight

    else:
        w = Multiply(name=w_name)([x, mask2])  # vec_heat
    return w




def vgg_block(x , weight_decay):
    # Block 1
    x = conv(x , 64 , 3 ,  "conv1_1", (weight_decay, 0))
    x = relu(x)
    x = conv(x , 64 , 3,    "conv1_2", (weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool1_1")
    
    # Block 2
    x = conv(x, 128, 3, "conv2_1", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "conv2_2", (weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool2_1")
    
    # Block 3
    x = conv(x, 256, 3, "conv3_1", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv3_2", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv3_3", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv3_4", (weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool3_1")
    
    # Block 4
    x = conv(x, 512, 3, "conv4_1", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 512, 3, "conv4_2", (weight_decay, 0))
    x = relu(x)
    
    # Additional non vgg layers
    x = conv(x, 256, 3, "conv4_3_CPM", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "conv4_4_CPM", (weight_decay, 0))
    x = relu(x)
    
    return x



def stage1_block(x, num_p, branch, weight_decay):
    # Block 1
    x = conv(x, 128, 3, "Mconv1_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "Mconv2_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "Mconv3_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 512, 1, "Mconv4_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, num_p, 1, "Mconv5_stage1_L%d" % branch, (weight_decay, 0))

    return x



def stageT_block(x, num_p, stage, branch, weight_decay):
    # Block 1
    x = conv(x, 128, 7, "Mconv1_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv2_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv3_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv4_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 7, "Mconv5_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 1, "Mconv6_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = relu(x)
    x = conv(x, num_p, 1, "Mconv7_stage%d_L%d" % (stage, branch), (weight_decay, 0))

    return x






