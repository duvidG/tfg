#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 12:20:48 2022

@author: dgiron
"""

import pandas as pd
import os
from astropy.io import fits
from src.errors import InputError


def read_data(file_name):
    """
    Function to read the data from a datafile

    Parameters
    ----------
    file_name : str
        path to the file.

    Raises
    ------
    InputError
        the file does not exist.

    Returns
    -------
    data : list
        list with all the images.
    header : astropy object
        information about the data.

    """
    if os.path.exists(file_name):
        data = []
        with fits.open(file_name) as f:
            for i in range(f[0].header['NAXIS3']):
                data.append(pd.DataFrame(f[0].data[i]))
            header = f[0].header
        return data, header
    else:
        raise InputError('Wrong file name/path')
         
        




        
        
