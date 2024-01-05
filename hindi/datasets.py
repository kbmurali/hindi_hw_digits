# coding: utf-8

# In[1]:


import warnings

warnings.simplefilter("ignore")

from typing import Callable, Optional

from torchvision import transforms

import torchvision.datasets as datasets

import urllib.request

import tarfile

import os


# In[2]:


def Digits( root, train: bool =True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False ):
    parent_folder = os.path.join( root, 'devanagari' )
    
    data_folder = os.path.join( parent_folder, 'train' ) if train else os.path.join( parent_folder, 'test' )
    
    if download:
        if not os.path.exists( data_folder ):
            os.makedirs( parent_folder )

            imgs_tar_url = 'https://github.com/kbmurali/hindi_hw_deep_gan/blob/main/hindi_hw_digits.tar.gz?raw=true'
    
            tar_file =  os.path.join( parent_folder, 'hindi_hw_digits.tar.gz' )
    
            urllib.request.urlretrieve( imgs_tar_url, tar_file )
    
            tar = tarfile.open( tar_file )
            
            # Extract all files to the current directory
            tar.extractall( parent_folder )
            
            # Close the tar file
            tar.close()
    
            #Delete the tar file
            if os.path.isfile( tar_file ):
                os.remove( tar_file )

    return datasets.ImageFolder( data_folder, transform=transform, target_transform=target_transform )

