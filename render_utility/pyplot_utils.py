import numpy as np
import matplotlib.pyplot as plt
from definitions import *
from matplotlib.colors import LinearSegmentedColormap

def plot_2d_function(f,trajectory=None,xlim=(-2.0,2.0),ylim=(-2.0,2.0),n=200):

    ax = plt.gca()
    xs = np.linspace(*xlim,n)
    ys = np.linspace(*ylim,n)
    X,Y = np.meshgrid(xs,ys)
    Z = f(X,Y)

    bg_cmap = LinearSegmentedColormap.from_list('render_utility_blues',
        [(1.0,1.0,1.0,1.0),blue_1,blue_2])
    ax.contourf(X,Y,Z,levels=30,cmap=bg_cmap)
    ax.contour(X,Y,Z,levels=6,colors='black',linewidths=2,alpha=0.5)

    if trajectory is not None:
        pts = np.asarray(trajectory)
        ax.plot(pts[:,0],pts[:,1],
           linewidth=5,                                                                                   
           linestyle='--',
           dash_capstyle='round',                                                                         
           color=(0,0,0),alpha=1)
        # Start and end markers
        ax.scatter(pts[0,0],pts[0,1],color=blue_2,
            s=220,edgecolor='white',linewidth=2,zorder=6)
        ax.scatter(pts[-1,0],pts[-1,1],color=red_2,
            s=220,edgecolor='white',linewidth=2,zorder=6)

    ax.set_aspect('equal')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
