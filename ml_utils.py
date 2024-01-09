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

import random
import os

import time

from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


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
                 wx_flt_index=0, 
                 wx_ch_index=0, 
                 wx_ro_index=0, 
                 wx_index=0,
                 wy_flt_index=0,
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
            wx = model[ conv_index ].weight[ wx_flt_index, wx_ch_index, wx_ro_index, wx_index ].item()
            wy = model[ conv_index ].weight[ wy_flt_index, wy_ch_index, wy_ro_index, wy_index ].item()
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

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

    def do_3d_projection( self, renderer=None ):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return np.min(zs)

def draw_loss_descent( results_df, figsize=(10,10), title=None ):
    wx = torch.tensor( results_df[ 'wx' ].values )
    wy = torch.tensor( results_df[ 'wy' ].values )
    ls = torch.tensor( results_df[ 'train loss' ].values )

    fig = plt.figure( figsize=figsize )
    ax = plt.axes(projection="3d")
    ax.scatter3D( wx.detach().cpu().numpy(), wy.detach().cpu().numpy(), ls.detach().cpu().numpy(), s=3 )

    for j in range( 1, wx.shape[0] ):
        i = j - 1
        Xs = [ wx[i].item(), wx[j].item() ]
        Ys = [ wy[i].item(), wy[j].item() ]
        Ls = [ ls[i].item(), ls[j].item() ]
        
        arw = Arrow3D( Xs, Ys, Ls, arrowstyle="->", color="purple", lw = 0.5, mutation_scale=9 )
        ax.add_artist(arw)

    if title is not None:
        ax.set_title(title, fontsize=10)

    ax.set_xlabel( 'Sample Weight 1' )
    ax.set_ylabel( 'Sample Weight 2' )
    ax.set_zlabel( 'Training Loss' )

    ax.xaxis.set_tick_params( labelbottom=False )
    ax.yaxis.set_tick_params( labelleft=False )
    ax.zaxis.set_tick_params( labelbottom=False )

    ax.set_box_aspect( aspect=None, zoom=0.8 )
    fig.tight_layout()
    plt.show()
    return fig