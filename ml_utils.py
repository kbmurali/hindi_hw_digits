import warnings

warnings.simplefilter("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import *

import torchvision
from torchvision import transforms
import torchvision.datasets as datasets

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm.autonotebook import tqdm

import pandas as pd

import time

def moveTo( obj, device ):
    if isinstance( obj, list ):
        return [moveTo(x,device) for x in obj]
    elif isinstance( obj, tuple ):
        return tuple( moveTo( list(obj), device ) )
    elif isinstance( obj, set ):
        return set( moveTo( list(obj), device ) )
    elif isinstance( obj, dict):
        to_ret = dict()

        for key, value in obj.items():
            to_ret[ moveTo( key, device) ] = moveTo( value, device )

        return to_ret
    elif hasattr( obj, "to" ):
        return obj.to( device )
    else:
        return obj

def visualize2DSoftmax2(X, y, model, title=None, device="cpu" ):
    x_min = np.min(X[:,0])-0.5
    x_max = np.max(X[:,0])+0.5
    y_min = np.min(X[:,1])-0.5
    y_max = np.max(X[:,1])+0.5
    
    xv, yv = np.meshgrid(np.linspace(x_min, x_max, num=20), np.linspace(y_min, y_max, num=20), indexing='ij')
    
    xy_v = np.hstack((xv.reshape(-1,1), yv.reshape(-1,1)))
    
    with torch.no_grad():
        logits = model(torch.tensor(xy_v, dtype=torch.float32, device=device))
        y_hat = F.softmax(logits, dim=1).cpu().numpy()
    
    cs = plt.contourf(xv, yv, y_hat[:,0].reshape(20,20), levels=np.linspace(0,1,num=20), cmap=plt.cm.RdYlBu )
    
    ax = plt.gca()
    
    sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, style=y, ax=ax)
    
    if title is not None:
        ax.set_title(title)


def run_epoch( model, optimizer, data_loader, loss_func, device="cpu", results=[], score_funcs={}, prefix=" ", desc=None ):
    running_loss = []
    y_true = []
    y_pred = []
    start = time.time()

    for inputs, labels in tqdm(data_loader, desc=desc, leave=False, disable=True ):
        inputs = moveTo( inputs, device )
        labels = moveTo( labels, device )

        if model.training:
            with torch.enable_grad():
                optimizer.zero_grad()
                y_hat = model( inputs )
                loss = loss_func( y_hat, labels )
                loss.backward()
                optimizer.step()
        else:
            with torch.no_grad():
                y_hat = model( inputs )
                loss = loss_func( y_hat, labels )

        running_loss.append( loss.item() )

        if len( score_funcs ) > 0 and isinstance( labels, torch.Tensor ):
            labels = labels.detach().cpu().numpy()
            y_hat = y_hat.detach().cpu().numpy()

            y_true.extend( labels.tolist() )
            y_pred.extend( y_hat.tolist() )
            
    end = time.time()

    y_pred = np.asarray(y_pred)
    
    if len(y_pred.shape) == 2 and y_pred.shape[1] > 1: #We have a classification problem, convert to labels
        y_pred = np.argmax(y_pred, axis=1)
    #Else, we assume we are working on a regression problem
    
    results[prefix + " loss"].append( np.mean(running_loss) )
    
    for name, score_func in score_funcs.items():
        try:
            results[prefix + " " + name].append( score_func(y_true, y_pred) )
        except:
            results[prefix + " " + name].append(float("NaN"))
            
    return end-start

def train_model( model,
                 loss_func, 
                 train_loader, 
                 test_loader=None, 
                 score_funcs=None, 
                 epochs=50, 
                 device='cpu', 
                 capture_conv_sample_weights=False, 
                 conv_index=0, 
                 wx_bt_index=0, 
                 wx_ch_index=0, 
                 wx_ro_index=0, 
                 wx_index=0,
                 wy_bt_index=0,
                 wy_ch_index=0, 
                 wy_ro_index=0, 
                 wy_index=1 ):
    results = {}

    results[ "epoch" ] = []
    results[ "total time" ] = []
    results[ "train loss" ] = []

    if capture_conv_sample_weights == True and isinstance( model, nn.Sequential ):
        results[ "wx" ] = []
        results[ "wy" ] = []

    if test_loader is not None:
        results[ "test loss" ] = []

    for eval_score in score_funcs:
        results[ "train " + eval_score ] = []
        if test_loader is not None:
             results[ "test " + eval_score ] = []

    total_train_time = 0
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
   
    model.to(device)

    for epoch in tqdm(range(epochs), desc="Epoch"):
        model.train()

        total_train_time += run_epoch( model, optimizer, train_loader, loss_func, device, results, score_funcs, prefix="train", desc="Training" )

        results["total time"].append( total_train_time )
        results["epoch"].append( epoch )

        if capture_conv_sample_weights == True and isinstance( model, nn.Sequential ):
            wx = model[ conv_index ].weight[ wx_bt_index, wx_ch_index, wx_ro_index, wx_index ].item()
            wy = model[ conv_index ].weight[ wy_bt_index, wy_ch_index, wy_ro_index, wy_index ].item()
            results["wx"].append( wx )
            results["wy"].append( wy )

        if test_loader is not None:
            model = model.eval()
            run_epoch( model, optimizer, test_loader, loss_func, device, results, score_funcs, prefix="test", desc="Testing" )

    return pd.DataFrame.from_dict( results )

def pred(model, img):
    with torch.no_grad():
        w, h = img.shape
        
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img)
        
        x = img.reshape(1,-1,w,h)
        
        logits = model(x)

        print( logits )
        
        y_hat = F.softmax(logits, dim=1)

        print( y_hat )
    
    return y_hat