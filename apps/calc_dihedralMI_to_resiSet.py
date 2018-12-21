# -*- coding: utf-8 -*-
# @Author: sukrit
# @Date:   2018-01-11 16:32:37
# @Last Modified by:   Sukrit Singh
# @Last Modified time: 2018-12-21 17:21:51

"""This apps script works to extract communication per residue to a target site
using a single holistic MI matrix. 
"""

import os
import numpy as np
from cardsReader.analysis import corrAnalysis as ca
import argparse

######################## 

parser = argparse.ArgumentParser(description="""
Calculation of all-dihedral correlations between residues
""")


parser.add_argument('-m','--matrix', default=os.getcwd(), 
                        help = "Path to .dat file containing holistic matrix")
parser.add_argument('-i','--indices', default=os.getcwd(), 
                        help = "Mapping file defining residue ID per MI matrix element")
parser.add_argument('-t','--topology', default=os.getcwd(), 
                        help = "Path to topology file of simulation set")
parser.add_argument('-r','--residues',  nargs='+', 
                        help="<Required> Indices of residues defining target site")
parser.add_argument('-o', '--output', default="MI_targetSite.dat", 
                        help = "Name of output file for target site communication")


####################### FUNCTIONS #############################

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
    #Collect Inputs 
    args = parser.parse_args()
    datafile = args.matrix
    pdbFile = args.topology
    resi_file = args.indices
    resiSet = args.residues
    outName = args.output


    data = main_method(datafile, pdbFile, resi_file, np.asarray(resiSet))
    np.savetxt(outName, data)


