#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 16:36:20 2022

@author: dgiron
"""

from src.classification import GalaxyDetector
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import pandas as pd
import numpy as np

def main():
    """
    Example of how to use the GalaxyDetector class. The biggest datset is 
    used for training the small one is used to make predictions.

    Returns
    -------
    None.

    """
    plt.close('all')

    start_time = time.time()
    
    # File names
    data_file = "/home/dgiron/tfg_data/full_dataset/cont_ldev.fits"
    data_file_ps = '/home/dgiron/tfg_data/full_dataset/sky_ldev_v2.fits'
    truth_cat = "/home/dgiron/tfg_data/full_dataset/sky_ldev_truthcat_v2.txt"
    
    data_file_small = "/home/dgiron/tfg_data/cont_dev.fits"
    data_file_ps_small = '/home/dgiron/tfg_data/sky_dev_v2.fits'
    truth_cat_small = '/home/dgiron/tfg_data/sky_dev_truthcat_v2.txt'
    
    # Common parameters
    dim = 64
    flujo_min = 12.5
    reduction_function = "mean_frec"
    overlapping = 'auto'
    
    # Create a object of the class GalaxyDetector

    file_size = 'b'
    if file_size == 'b':
        train = GalaxyDetector(data_file, dim, overlapping, reduction_function, 
                               truth_cat, data_file_ps, 
                               min_major_length=0, min_minor_length=0, 
                               flujo_min=flujo_min, norm='max')
    if file_size == 's':
        train = GalaxyDetector(data_file_small, dim, overlapping, reduction_function, 
                               truth_cat_small, data_file_ps_small,
                               min_major_length=0, min_minor_length=0,
                               flujo_min=flujo_min, norm='max')
        
        
    
    # Plot the mask of the datacube number 0 
    train.plot_datacube_label(0)
    
    # Draw the image of the small datacube number 0
    train.plot_small_reduced_datacube(0) 
    
    # Draw the original dataset, after the reduction of dimensions
    train.plot_full_reduced_datacube()
    
    # Plot histogram of the sources, attending to its integrated flux
    train.plot_flux_histogram(20) 
    
    # Prueba NN
        # Hyperparameters
    fraction_for_training = 0.8
    threshold = 0.035
    batch_size = 32
    epochs = 200
    
    # Datacube to test the net
    file_size = 's'
    if file_size == 'b':
        test = GalaxyDetector(data_file, dim, overlapping, reduction_function, 
                              truth_cat, data_file_ps, 
                              min_major_length=0, min_minor_length=0, 
                              flujo_min=flujo_min, norm='max')
    if file_size == 's':
        test = GalaxyDetector(data_file_small, dim, overlapping, reduction_function, 
                              truth_cat_small, data_file_ps_small, 
                              min_major_length=0, min_minor_length=0, 
                              flujo_min=flujo_min, norm='max')
        
    
    # Learning rate function
    initial_learning_rate = 0.1
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)
    
    
    # Method to train the N
    train.solve_NN(fraction_for_training, dim, 
                    batch_size, epochs, filt='auto', optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
                    architecture='first_model', save_scheme=True)
    
    # Make predictions with the normalized images of the small dataset, saving the catalogue in a DataFrame
    final_catalogue = pd.DataFrame([], columns= ['ra', 'dec', 'hi_size', 'i', 'pa'])
    for i in range(0, test.length_list_minicubes, 1):
        new_catalogue = train.prediction(dim='auto', img=test.small_datacubes_norm[i], real_mask=test.all_total_labels[i], 
                          img_num=i, truth_cat_small=test.truth_cat_small[i], thr=threshold)
        final_catalogue = pd.concat([final_catalogue, new_catalogue], ignore_index=True)
        
                
    # Prepare the catalogue to be saved in a csv file
    final_catalogue.index.name = 'id'
    
    
    final_catalogue = final_catalogue.apply(pd.to_numeric)

    final_catalogue.to_csv('src/output/final_catalogue.csv')

    truth_cat = test.truth_cat_original
    truth_cat.index.name = 'id'
    truth_cat.to_csv("src/output/truth_catalogue.csv")
    
    # Evaluate the model
    matched_cat, stats = GalaxyDetector.score2(final_catalogue, truth_cat, 20)
    
    print(stats)
    
    print("--- %s seconds ---" % (time.time() - start_time))

main()

    