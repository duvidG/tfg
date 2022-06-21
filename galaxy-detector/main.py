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
    plt.ioff()
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
    # flujo_min = 0
    flujo_min = 12.5
    reduction_function = "mean_frec"
    overlapping = 'auto'


    file_size = 'b'
    if file_size == 'b':
        train = GalaxyDetector(data_file, dim, overlapping, reduction_function, 
                               truth_cat, data_file_ps, 
                               min_major_length=0, min_minor_length=0, 
                               flujo_min=flujo_min)
    if file_size == 's':
        train = GalaxyDetector(data_file_small, dim, overlapping, reduction_function, 
                               truth_cat_small, data_file_ps_small,
                               min_major_length=0, min_minor_length=0,
                               flujo_min=flujo_min)
    
    # Imprimir figs intermedias [409, 455, 472, 509, 539, 612, 619, 644, 667, 680, 689, 696]
    # for i in range(0, train.length_list_minicubes):
    #     if i in [115]:
    #         train.plot_datacube_label(i)
    #         plt.savefig('../informe/imgs/resultados_label_'+str(i)+'.png')
    #         # train.plot_small_reduced_datacube(i)
    #         # plt.savefig('imgs/resultados_minicube_prueba'+str(i)+'.png')
    #         plt.close('all')
        # train.plot_small_reduced_datacube(i)
    # train.plot_small_reduced_datacube(-1)
    # train.plot_full_reduced_datacube()
    # train.plot_flux_histogram(20)
    
    # Prueba NN
        # Hyperparameters
    fraction_for_training = 0.8
    threshold = 0.035
    batch_size = 32
    epochs = 200
    
    file_size = 's'
    if file_size == 'b':
        test = GalaxyDetector(data_file, dim, overlapping, reduction_function, 
                              truth_cat, data_file_ps, 
                              min_major_length=0, min_minor_length=0, 
                              flujo_min=flujo_min)
    if file_size == 's':
        test = GalaxyDetector(data_file_small, dim, overlapping, reduction_function, 
                              truth_cat_small, data_file_ps_small, 
                              min_major_length=0, min_minor_length=0, 
                              flujo_min=flujo_min)
        
    
    initial_learning_rate = 0.1
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)
    
    
    
    train.solve_NN(fraction_for_training, dim, 
                    batch_size, epochs, filt='auto', optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
                    architecture='swin_unet_2d', save_scheme=False)
    final_catalogue = pd.DataFrame([], columns= ['ra', 'dec', 'hi_size', 'i', 'pa'])
    for i in range(0, test.length_list_minicubes, 1):
        new_catalogue = train.prediction(dim='auto', img=test.small_datacubes[i], real_mask=test.all_total_labels[i], 
                          img_num=i, truth_cat_small=test.truth_cat_small[i], thr=threshold)
        
        final_catalogue = pd.concat([final_catalogue, new_catalogue], ignore_index=True)
        
        
        # plt.close('all')

    
    final_catalogue.index.name = 'id'
    final_catalogue['line_flux_integral'] = np.mean(test.truth_cat_original['line_flux_integral'])
    final_catalogue['central_freq'] = np.mean(test.truth_cat_original['central_freq'])
    final_catalogue['w20'] = np.mean(test.truth_cat_original['w20'])
    cols = ['ra', 'dec', 'hi_size', 'line_flux_integral', 'central_freq', 'pa', 'i', 'w20']
    final_catalogue = final_catalogue[cols]
    
    
    
    
    # final_catalogue.index[0] = 'id'
    final_catalogue = final_catalogue.apply(pd.to_numeric)

    final_catalogue.to_csv('src/output/final_catalogue.csv')

    truth_cat = test.truth_cat_original
    truth_cat.index.name = 'id'
    truth_cat.to_csv("src/output/truth_catalogue.csv")
    # score = GalaxyDetector.score('src/output/final_catalogue.csv', "src/output/truth_catalogue.csv")
    
    print("--- %s seconds ---" % (time.time() - start_time))

main()

    