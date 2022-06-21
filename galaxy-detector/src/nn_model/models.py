#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 19:00:25 2022

@author: dgiron
"""

import tensorflow as tf
from keras.layers import Input, Activation
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate

from keras_unet_collection import models as segmentation_models # https://github.com/yingkaisha/keras-unet-collection



def first_model(input_size, n_filters, n_classes):
    def convolutional_block(inputs=None, n_filters=4, dropout_prob=0, max_pooling=True, iteration=0):
        """
        Layer of the contractive part of the model

        Parameters
        ----------
        inputs : tf.model, optional
            output from the previous layer. The default is None.
        n_filters : int, optional
            number of filters for the first layer. The default is 4.
        dropout_prob : float, optional
            a number bigger than zero activates the Dropout layer. Its use is recommended
            to prevent overfitting. The default is 0.
        max_pooling : bool, optional
            if True  halves the dimensions at the end of the layer. The default is True.
        iteration : int, optional
            number of level. The kernel_size is almost four times bigger in the first two
            layers than for the rest. The default is 0.

        Returns
        -------
        next_layer : TYPE
            DESCRIPTION.
        skip_connection : TYPE
            DESCRIPTION.

        """
        if iteration < 3:
            ker_size = 11
        else:
            ker_size = 3
        conv = Conv2D(n_filters, 
                      kernel_size = ker_size,
                      activation='relu',
                      padding='same',
                      kernel_initializer=tf.keras.initializers.HeNormal())(inputs)
        
        conv = Conv2D(n_filters, 
                      kernel_size = ker_size,
                      activation='relu',
                      padding='same',
                      kernel_initializer=tf.keras.initializers.HeNormal())(conv)
       
    
    
        if dropout_prob > 0:
            conv = Dropout(dropout_prob)(conv)
            
        if max_pooling:
            next_layer = MaxPooling2D((2, 2), padding='same')(conv)
        else:
            next_layer = conv
    
        #conv = BatchNormalization()(conv)
        skip_connection = conv
        
        return next_layer, skip_connection
    
    def upsampling_block(expansive_input, contractive_input, n_filters=4):
        """
        Layer of the expansive part of the network

        Parameters
        ----------
        expansive_input : tf.layer
            output from the previous layer.
        contractive_input : tf.layer
            output from the opposite layer, in the contractive part.
        n_filters : int, optional
            number of filters for the last layer. The default is 4.

        Returns
        -------
        conv : TYPE
            DESCRIPTION.

        """
            
        up = Conv2DTranspose(
                     n_filters,  
                     kernel_size = 3,
                     strides=(2,2),
                     padding='same')(expansive_input)
        
        merge = concatenate([up, contractive_input], axis=3)
        conv = Conv2D(n_filters,  
                     kernel_size = 3,   
                     activation='relu',
                     padding='same',
                     kernel_initializer=tf.keras.initializers.HeNormal())(merge)
        conv = Conv2D(n_filters,  
                     kernel_size = 3,  
                     activation='relu',
                     padding='same',
                     kernel_initializer=tf.keras.initializers.HeNormal())(conv)
        
        return conv
    
    
    
    def unet_model(input_size, n_filters=4, n_classes=3):
        """
        U-net model constructed from 'scratch'

        Parameters
        ----------
        input_size : tuple
            size of the images, consisting in three elements (2xregular size + 3rd dimension) For example,
            a RGB would have 3 as the third dimension. On the other hand, a gray-scale image would have 1.
        n_filters : int, optional
            number of filters for the first layer. The default is 4.
        n_classes : TYPE, optional
            total number of labels. The default is 3.

        Returns
        -------
        model : tensorflow.model
            final model.

        """
    
        inputs = Input(input_size)
        # tf.keras.layers.Rescaling(1/maximo, offset=0.0)

        
        #contracting path
        cblock1 = convolutional_block(inputs, n_filters, iteration=1)
        cblock2 = convolutional_block(cblock1[0], 2*n_filters, iteration=2)
        cblock3 = convolutional_block(cblock2[0], 4*n_filters, iteration=3)
        cblock4 = convolutional_block(cblock3[0], 8*n_filters, iteration=4, dropout_prob=0.2) 
        cblock5 = convolutional_block(cblock4[0],16*n_filters, iteration=5, dropout_prob=0.2, max_pooling=None)     
        
        #expanding path
        ublock6 = upsampling_block(cblock5[0], cblock4[1],  8 * n_filters)
        ublock7 = upsampling_block(ublock6, cblock3[1],  n_filters*4)
        ublock8 = upsampling_block(ublock7, cblock2[1] , n_filters*2)
        ublock9 = upsampling_block(ublock8, cblock1[1],  n_filters)
    
        conv9 = Conv2D(n_classes,
                       1,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(ublock9)
        
        #conv10 = Conv2D(n_classes, kernel_size=1, padding='same', activation = 'softmax')(conv9) 
        conv10 = Activation('softmax')(conv9)
    
        model = tf.keras.Model(inputs=inputs, outputs=conv10)
    
        return model
    
    return unet_model(input_size, n_filters, n_classes)



# #def first_model(input_size, n_filters, n_classes):
#     def convolutional_block(inputs=None, n_filters=4, dropout_prob=0, max_pooling=True):
#         conv = Conv2D(n_filters, 
#                       kernel_size = 3,
#                       activation='relu',
#                       padding='same',
#                       kernel_initializer=tf.keras.initializers.HeNormal())(inputs)
        
#         conv = Conv2D(n_filters, 
#                       kernel_size = 3,
#                       activation='relu',
#                       padding='same',
#                       kernel_initializer=tf.keras.initializers.HeNormal())(conv)
       
    
    
#         if dropout_prob > 0:
#             conv = Dropout(dropout_prob)(conv)
            
#         if max_pooling:
#             next_layer = MaxPooling2D((2, 2), padding='same')(conv)
#         else:
#             next_layer = conv
    
#         #conv = BatchNormalization()(conv)
#         skip_connection = conv
        
#         return next_layer, skip_connection
    
#     def upsampling_block(expansive_input, contractive_input, n_filters=4):
            
#         up = Conv2DTranspose(
#                      n_filters,  
#                      kernel_size = 3,
#                      strides=(2,2),
#                      padding='same')(expansive_input)
        
#         merge = concatenate([up, contractive_input], axis=3)
#         conv = Conv2D(n_filters,  
#                      kernel_size = 3,   
#                      activation='relu',
#                      padding='same',
#                      kernel_initializer=tf.keras.initializers.HeNormal())(merge)
#         conv = Conv2D(n_filters,  
#                      kernel_size = 3,  
#                      activation='relu',
#                      padding='same',
#                      kernel_initializer=tf.keras.initializers.HeNormal())(conv)
        
#         return conv
    
    
    
#     def unet_model(input_size, n_filters=4, n_classes=3):
    
#         inputs = Input(input_size)
        
#         #contracting path
#         cblock1 = convolutional_block(inputs, n_filters)
#         cblock2 = convolutional_block(cblock1[0], 2*n_filters)
#         cblock3 = convolutional_block(cblock2[0], 4*n_filters)
#         cblock4 = convolutional_block(cblock3[0], 8*n_filters, dropout_prob=0.2) 
#         cblock5 = convolutional_block(cblock4[0],16*n_filters, dropout_prob=0.2, max_pooling=None)     
        
#         #expanding path
#         ublock6 = upsampling_block(cblock5[0], cblock4[1],  8 * n_filters)
#         ublock7 = upsampling_block(ublock6, cblock3[1],  n_filters*4)
#         ublock8 = upsampling_block(ublock7, cblock2[1] , n_filters*2)
#         ublock9 = upsampling_block(ublock8, cblock1[1],  n_filters)
    
#         conv9 = Conv2D(n_classes,
#                        1,
#                        activation='relu',
#                        padding='same',
#                        kernel_initializer='he_normal')(ublock9)
        
#         #conv10 = Conv2D(n_classes, kernel_size=1, padding='same', activation = 'softmax')(conv9) 
#         conv10 = Activation('softmax')(conv9)
    
#         model = tf.keras.Model(inputs=inputs, outputs=conv10)
    
#         return model
    
#     return unet_model(input_size, n_filters, n_classes)
