# -*- coding: utf-8 -*-
# @Author: sukrit
# @Date:   2018-01-11 16:32:37
# @Last Modified by:   Sukrit Singh
# @Last Modified time: 2018-12-27 13:35:05

"""This apps script works to extract communication per residue to a target site
using a single holistic MI matrix. 
"""

import os
import numpy as np
from cardsReader.analysis import corrAnalysis as ca
import argparse
import matplotlib.pyplot as plt
import mdtraj as md

############ PLOTTING PARAMS ###########
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 17
plt.rcParams['axes.labelsize'] = 17
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 17
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['figure.titlesize'] = 12

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

def convert_inds_to_resis(inds, pdbFile):
    traj = md.load(pdbFile)
    resi_ids = np.zeros(inds.shape[0])
    for i,n in enumerate(inds):
        atomVal = int(n[1])
        resi_val = traj.top.atom(atomVal).residue.resSeq
        resi_ids[i] = resi_val

    return resi_ids

def main_method(datafile, pdbFile, resi_file, resiSet):
    # Process dihedal indices input file 
    allInds = np.loadtxt(resi_file, delimiter=',')
    allResis = convert_inds_to_resis(allInds, pdbFile)
    resis = np.unique(allResis)
    data = np.loadtxt(datafile)

    # Neighbor using method
    # Mean-Based with boot-strapping for error 
    corrAmount = ca.compute_dihedral_based_MI_to_resiSet_error(
        data, resiSet, allResis, pdbFile, 3.0)

    #resis = np.unique(allResis)
    final = np.zeros((resis.shape[0], 3))
    final[:, 0] = resis
    final[:, 1] = corrAmount[:,0]
    final[:, 2] = corrAmount[:,1]

    return final


if __name__ == '__main__':
    #Collect Inputs 
    args = parser.parse_args()
    datafile = args.matrix
    pdbFile = args.topology
    resi_file = args.indices
    resiSet = args.residues
    outName = args.output


    data = main_method(datafile, pdbFile, resi_file, np.asarray(resiSet).astype(float))
    np.savetxt(outName, data)

    # Make a plot of the data
    fig = plt.figure()
    plt.errorbar(x=data[:,0], y=data[:,1], yerr=data[:,2], lw=1.2, capsize=1.5)
    plt.xlabel("Residue number")
    plt.ylabel("MI to target site")
    plt.xlim(data[0,0], data[-1:,0])
    fig.set_size_inches(11,7)
    plt.tight_layout()
    plt.savefig(outName[:-4]+'.png', dpi=300)

    #plt.show()




