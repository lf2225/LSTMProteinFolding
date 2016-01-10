# LSTMProteinFolding
Had the idea of using LSTM RNNs to predict protein folding. The way I did this was consider protein folding as a linear transformation from R^3 to R^3 of atom coordinates. I first arrange the atoms of an amino acid in their relative positions accoording to their PDB files, and then arrange the amino acids in a straight line on the x-axis. These are the inputs for a single protein, and the output is the atom coordinates from the folded protein's PDB file. 


