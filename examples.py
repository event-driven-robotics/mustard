# -*- coding: utf-8 -*-
"""
Copyright (C) 2020 Event-driven Perception for Robotics
Authors: Massimiliano Iacono
         Sim Bamford

This program is free software: you can redistribute it and/or modify it under 
the terms of the GNU General Public License as published by the Free Software 
Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY 
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with 
this program. If not, see <https://www.gnu.org/licenses/>.

Examples of how to use "mustard" visualiser to visualise imported data.
It's possible to use the visualizer as a standalone app, 
using its load dialog to get data into it. 
However this script assumes running the code section by section 
(sections delimited by #%%) as in an IDE such as pycharm or spyder. 
The visualiser is started in a thread, then data is pushed to it from the IDE.
This allows for a workflow in which data is progressively imported and manipulated
in the IDE, and inspected by the visualizer at each stage. 
Note, however, that if the visualiser is closed or crashes, it must be re-run 
from a fresh console, which may mean re-importing and processing data. 
"""

#%% Preliminaries:

''' 
Run this from a fresh console
'''

import os, sys
import threading

# If you haven't installed bimvee or mustard as packages, set up your local paths
# Get os-specific path for local libraries (YMMV)
prefix = 'C:/' if os.name == 'nt' else '/home/sbamford/'    
sys.path.append(os.path.join(prefix, 'repos/bimvee'))

from mustard.mustard import Mustard
# Create the mustard  app and start it in a thread
app = Mustard()
thread = threading.Thread(target=app.run)
thread.daemon = True
thread.start()

# !! Wait until the load dialog has come up, otherwise the component that
# receives your data won't have been created yet

#%% Load some data that you want to work with

from bimvee.importIitYarp import importIitYarp

filePathOrName = os.path.join(prefix, "data/2019_11_11_AikoImu/linear_100/ATIS")
container = importIitYarp(filePathOrName=filePathOrName, tsBits=30)

# Having loaded a dvs dataDict - poke it into the right place

# Firstly clear it - if you edit a dict and pass it in again it won't trigger 
# the change event as it will be the same object
app.root.data_controller.data_dict = {}  
app.root.data_controller.data_dict = container

#%% Load some secdvs data

from bimvee.importSecDvs import importSecDvs

filePathOrName = os.path.join(prefix, "data/2020_03_23 SecDvs from Ander/2020-03-24-12-45-13.bin")
container = importSecDvs(filePathOrName=filePathOrName)

app.root.data_controller.data_dict = {}
app.root.data_controller.data_dict = container

#%% Load some generic data

from bimvee.importAe import importAe

filePathOrName = os.path.join(prefix, "data/2020_03_23 SecDvs from Ander/2020-03-24-12-45-13.bin")
container = importAe(filePathOrName=filePathOrName)

app.root.data_controller.data_dict = {}
app.root.data_controller.data_dict = container

#%% Simulated DAVIS data

# This example demonstrates the use of a template to limit what data is imported,
# and therefore what data gets visualised. 
# You can also create a container with just the data you want before passing 
# it to the visualiser.

# http://rpg.ifi.uzh.ch/davis_data.html
from bimvee.importRpgDvsRos import importRpgDvsRos
    
filePathOrName = os.path.join(prefix, 'data/rpg/simulation_3walls.bag')

container = importRpgDvsRos(filePathOrName=filePathOrName)

app.root.data_controller.data_dict = {}
app.root.data_controller.data_dict = container

#%% Experiment with pose interpolation

pose = container['data']['extra']['pose6q']
toKeep = [0, 300, 600, 900, 1200, 1500, 1800, 1999]
poseKept = {} 
poseKept['ts'] = pose['ts'][toKeep]
poseKept['point'] = pose['point'][toKeep, :]
poseKept['rotation'] = pose['rotation'][toKeep, :]

container['data']['reduced'] = {'pose6q': poseKept}

app.root.data_controller.data_dict = {}
app.root.data_controller.data_dict = container

#%% Constructed rolling poses - X

import numpy as np

ts = np.array([0, 1, 2, 3, 4], dtype=np.float64)
point = np.zeros((5, 3), dtype=np.float64)
# from neutral pose, to x = 1 (i.e. roll backwards through 180) 
# back to neutral, then x to -1 ( i.e. roll forwards head over heels through 180)
# back to neutral 
rotation = np.array([[1, 0, 0, 0], 
                     [0, 1, 0, 0], 
                     [1, 0, 0, 0], 
                     [0, -1, 0, 0], 
                     [1, 0, 0, 0]],
                    dtype=np.float64)

container = {'pose6q': {'ts': ts,
                        'point': point,
                        'rotation': rotation
                        }
            }

app.root.data_controller.data_dict = {}
app.root.data_controller.data_dict = container

#%% Apply bounding boxes to frames and events

from bimvee.importAe import importAe

filePathOrName = os.path.join(prefix, 'data/2020_05_11_Massi_BoundingBox/numpy/events.npy')

container = importAe(filePathOrName=filePathOrName)

app.root.data_controller.data_dict = {}
app.root.data_controller.data_dict = container
