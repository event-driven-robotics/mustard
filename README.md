# MUSTARD
MUlti STream Agnostic Representation Dataplayer

This GUI can be used to visualize multiple streams of data encoded in any format supported by BIMVEE, which is the data management library used under the hood. The main purpose is to visualize event-based data, alongside with any accompanying stream being RGB frames, IMU, skeleton for human pose estimation. It also features an annotation and visualization tool to draw bounding boxes and eye gaze on vision-derived data.

# Install via PIP

```
pip install mustard-gui
```

# Install from source

Clone this repo, open a terminal in the cloned directory and then run 
```
python setup.py install
```

You might need to have administrator privileges.

# Run the GUI

If you have installed using any of the above method you can run
```
mustard
```
or 
```
python -m mustard
```
