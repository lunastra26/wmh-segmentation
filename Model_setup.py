"""
CNN architecture and loss functions 
Author: LU
Feb 2021
Umapathy, L., G. G. Perez-Carrillo, M. B. Keerthivasan, J. A. Rosado-Toro, M. I. Altbach, B. Winegar, 
C. Weinkauf, A. Bilgin, and Alzheimer’s Disease Neuroimaging Initiative. 
"A Stacked Generalization of 3D Orthogonal Deep Learning Convolutional Neural Networks for Improved Detection 
of White Matter Hyperintensities in 3D FLAIR Images." American Journal of Neuroradiology (2021).
DOI: https://doi.org/10.3174/ajnr.A6970
"""

import numpy as np 
import sys, os
import tensorflow as tf
from keras import backend as K
from keras import optimizers
from functools import partial, update_wrapper
from keras.models import Model, load_model
from keras.layers import Input, Dropout, concatenate, BatchNormalization,Activation
from keras.layers.convolutional import Conv3D, MaxPooling3D, Conv3DTranspose
 

def setTF_environment(visible_gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = visible_gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    config.intra_op_parallelism_threads = 1
    config.inter_op_parallelism_threads = 1
    K.set_session(tf.Session(config=config))
    return
   
def wrapped_partial(func, *args, **kwargs):
	partial_func = partial(func, *args, **kwargs)
	update_wrapper(partial_func, func)
	return partial_func

def WBCE_custom_loss(y_true, y_pred, class_weights):
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    loss = K.mean(class_weights * (-y_true * K.log(y_pred) - (1.0 - y_true) * K.log(1.0 - y_pred)),axis=-1)
    return loss

def loadSavedModel(modelLoadPath, class_weights):
    model = load_model(modelLoadPath)
    Adamopt = optimizers.Adam(lr = 1e-3)
    wbce_loss = wrapped_partial(WBCE_custom_loss,class_weights=np.asarray(class_weights))
    model.compile(loss=wbce_loss, optimizer=Adamopt)
    return model

def initializeMetaModel(ipShape,modelSavePath):
    AdamOpt = optimizers.SGD(lr = 0.001)
    model = metaCNN(input_width=ipShape[0],input_height=ipShape[1],input_slice=ipShape[2],nChannels=3,num_classes=2)  
    model.load_weights(modelSavePath)
    model.compile(optimizer=AdamOpt, loss='categorical_crossentropy') 
    return model

def metaCNN(input_width=16,input_height=16,input_slice=16,nChannels=3,num_classes=2):
    from keras.layers import Conv3D, Input
    from keras.models import Model
    inputs = Input((input_width,input_height,input_slice,nChannels))
    y = Conv3D(2,(1,1,1),padding='same',activation='softmax')(inputs)
    model = Model(inputs=[inputs], outputs=[y])
    return model

def conv_block_encode(block_input,numFt1,numFt2,conv_kernel,upsample_flag=True,pool_size=(2,2,1)):
    # Defining a Convolutional block in the encoding path of UNET    
    if(upsample_flag):
        block_input = MaxPooling3D(pool_size=pool_size)(block_input)        
    down = Conv3D(numFt1,conv_kernel,padding='same')(block_input)
    down = BatchNormalization()(down)
    down = Activation('relu')(down)
    down = Conv3D(numFt1,conv_kernel,padding='same')(down)
    down = BatchNormalization()(down)
    down = Activation('relu')(down)
    down = Dropout(0.5)(down)
    down = Conv3D(numFt2,conv_kernel,padding='same')(block_input)
    down = BatchNormalization()(down)
    down = Activation('relu')(down)
    down = Conv3D(numFt2,conv_kernel,padding='same')(down)
    down = BatchNormalization()(down)
    down = Activation('relu')(down) 
    return down
 
def conv_block_decode(block_input,numFt1,numFt2,concat_block,conv_kernel,strides=(2,2,1)):
    # Defining a Convolutional block in the decoding path of UNET
    up = Conv3DTranspose(numFt1,(3,3,3), strides=strides, padding='same')(block_input)
    up = concatenate([up, concat_block], axis=4)
    up = Conv3D(numFt1,conv_kernel,padding='same')(up)
    up = BatchNormalization()(up)
    up = Activation('relu')(up)
    up = Conv3D(numFt1,conv_kernel,padding='same')(up)
    up = BatchNormalization()(up)
    up = Activation('relu')(up)
    up = Dropout(0.5)(up)
    up = Conv3D(numFt2,conv_kernel,padding='same')(up)
    up = BatchNormalization()(up)
    up = Activation('relu')(up)
    up = Conv3D(numFt2,conv_kernel,padding='same')(up)
    up = BatchNormalization()(up)
    up = Activation('relu')(up)   
    return up

def DeepUNET3D(input_width=64,input_height=64,input_slice=5,nChannels=1,num_classes=1,conv_kernel=(3,3,3),final_Activation='softmax'):
    inputs = Input((input_width,input_height,input_slice,nChannels))
    # Contracting path (Encoding)
    down_blk1 = conv_block_encode(inputs,8,16,conv_kernel,upsample_flag=False)
    down_blk2 = conv_block_encode(down_blk1,16,32,conv_kernel)
    down_blk3 = conv_block_encode(down_blk2,32,64,conv_kernel)
    center_blk4 = conv_block_encode(down_blk3,64,128,conv_kernel)  
    center_blk4 = Dropout(0.5)(center_blk4)
    # Expanding path (Decoding)
    up_blk3 = conv_block_decode(center_blk4,64,32,down_blk3,conv_kernel)
    up_blk2 = conv_block_decode(up_blk3,32,16,down_blk2,conv_kernel)
    up_blk1 = conv_block_decode(up_blk2,16,8,down_blk1,conv_kernel)
    posterior = Conv3D(num_classes, (1,1,1), activation=final_Activation)(up_blk1)
    model = Model(inputs=[inputs], outputs=[posterior])
    return model
 


