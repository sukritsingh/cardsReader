# -*- coding: utf-8 -*-
# @Author: sukrit
# @Date:   2018-01-11 16:32:37
# @Last Modified by:   Sukrit Singh
# @Last Modified time: 2018-11-02 17:00:51


import numpy as np
from cardsReader.analysis import corrAnalysis as ca


def main_method(datafile, pdbFile, resi_file, resiSet):
    allResis = np.loadtxt(resi_file)
    data = np.loadtxt(datafile)

    # Neighbor using method
    # Mean-Based with boot-strapping for error 
    corrAmount = ca.compute_dihedral_based_MI_to_resiSet_error(
        data, resiSet, allResis, pdbFile, 3.0)

    resis = np.unique(allResis)
    final = np.zeros((resis.shape[0], 3))
    final[:, 0] = resis
    final[:, 1] = corrAmount[:,1]
    final[:, 2] = corrAmount[:,2]

    return final

if __name__ == '__main__':
    # Provide the path to holistic matrix
    datafile = "/home/sukrit/work/gq/10195_gq_gdp_v/cards/data/fracMI_Hol_AllDihedrals.dat"
    # Provide path to PDB file
    pdbfile = "/home/sukrit/work/gq/10195_gq_gdp_v/cards/data/protein_2.pdb"
    # Provide path to mapping file (MAKE SURE IT IS RESIDUE-BASED)
    resi_file = "/home/sukrit/work/gq/10195_gq_gdp_v/cards/data/gq_10195_resSeq_Mapping.dat"

    # Residue set of interest
    #setOfinterest =  np.linspace(6, 334, 338)
    resiSet = np.asarray([56,60,67,75,78,185,187,190,192,193]) 
    

    # Aromatics
    #resiSet = [454, 482, 492, 560, 579]
    #[220+221+224+225+244+276+279+280+283+286]#
    data = main_method(datafile, pdbfile, resi_file, np.asarray(resiSet))
    np.savetxt("gq_10195_holMI_normalized_yMBindingSite_wError.dat", data)
