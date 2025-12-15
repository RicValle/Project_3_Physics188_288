# Classifying underlying lattice from random walks, using individual trajectory data

Task here: predict dilution percentage, lattice type, and whether hopping is short- or long-ranged, using information from individual random walker trajectories.

Uses a fully-connected feedforward neural network for each of the three lattice setups, and has a hand-designed featurization stage to convert arbitrary-length trajectories into a fixed-length feature vector.

Files:
* `pack_data.ipynb` to collate all raw data into a compact form for loading/saving
* `featurization.ipynb` to turn the trajectories into features
* `fcn.ipynb` to actually train and evaluate the network
