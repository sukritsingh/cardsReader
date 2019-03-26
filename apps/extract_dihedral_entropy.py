# -*- coding: utf-8 -*-
# @Author: Sukrit Singh
# @Date:   2018-11-02 16:14:22
# @Last Modified by:   Sukrit Singh
# @Last Modified time: 2019-03-26 10:52:56

"""
This script extracts the Shannon entropy of dihedral motion from the CARDS results
and saves them as a separate pickle file
"""

#################### IMPORTS #################
import mdtraj as md 
import os
import numpy as np
import argparse
import pickle
import pandas as pd


################# INPUT PARSER #####################
parser = argparse.ArgumentParser(description="""
Calculation of Shannon entropy per residue
""")


parser.add_argument('-m','--matrices', default=os.getcwd(), help = "Path to pickle file containing CARDS matrices")
parser.add_argument('-i','--indices', default=os.getcwd(), help = "Path to file containing dihedral indices")
parser.add_argument('-t','--topology', default=os.getcwd(), help = "Path to topology file of simulation set")


################### METHODS ####################

def extract_entropy_from_matrices(fileName):
    """
    This residue reads in the structural and dynamical correlation matrices provided
    """
    with open(fileName, 'rb') as f: 
        matrices = pickle.load(f)

    ss_mi = matrices['Struc_struc_MI']
    dd_mi = matrices['Disorder_disorder_MI']
    ds_mi = matrices['Disorder_struc_MI']
    sd_mi = matrices['Struc_disorder_MI'] 

    entropy_vals = np.diagonal(ss_mi)

    return entropy_vals


def get_resi_mapping(inds_file, top_File):
    """Given the indices array and an MDTraj object of the topology, 
    this method will output a mapping file identifiying which residue INDEX
    each set of indices belongs to. 

    Keep in mind this is NOT the residue sequence number (which can repeat in 
    multimers), but rather the index according to the topology. 
    """
    inds = np.loadtxt(inds_file, delimiter=",")
    structure = md.load(top_File)
    n_resis = structure.top.n_residues
    n_dihedrals = inds.shape[0]
    resi_map = np.zeros(n_dihedrals)
    for i, n in enumerate(inds):
        atom_index = n[1]
        resi_index = structure.top.atom(int(atom_index)).residue.index
        resi_map[i] = resi_index

    return resi_map

def sum_entropies_by_residue(resi_map, entropies):
    """This function will take the mapping produced in get_resi_mapping and 
    sum dihedral entropies up by residue to produce a normalized entropy per 
    residue. 
    """
    unique_resi_list = np.unique(resi_map)
    n_resis = unique_resi_list.shape[0]
    # if (n_resis != structure.top.n_residues):
    #     print("WARNING: The number of protein residues you have and \n")
    #     print("the number of residues in the topology do not match. \n")
    #     print("This is possible when there are ligands around, but please confirm \n")
    #     print("that this is okay.")
    entropies_per_residue = np.zeros(n_resis)
    for i, r in enumerate(unique_resi_list):
        residue_dihedral_values = entropies[resi_map == r]
        n_dihedrals = residue_dihedral_values.shape[0]
        residue_entropy = np.sum(residue_dihedral_values)/n_dihedrals
        entropies_per_residue[i] = residue_entropy

    return entropies_per_residue

def save_dihedral_entropy(entropy_values, resi_map, output_name):
    """Save dihedral entropies as a pandas dataframe csv containing information about
    each residue's entropy.
    """
    dihedral_output_array = np.vstack((resi_map, entropy_values)).T
    pandas_out_dihedrals = pd.DataFrame(dihedral_output_array)
    pandas_out_dihedrals.columns = ['residue index', 'dihedral entropy value']
    pandas_out_dihedrals.to_csv(output_name, index=False)

    print("Saved dihedral entropies as "+str(output_name))

    return 0 

def get_residue_seqIds(resi_list, top_file):
    structure = md.load(top_file)
    resSeq_list = np.zeros(resi_list.shape[0])
    for i, n in enumerate(resi_list):
        resSeq_list[i] = structure.top.residue(int(n)).resSeq

    return resSeq_list

def save_residue_entropy(entropy_residues, resi_map, structure, output_name):
    """Given a unique list of residues 
    """
    resi_list = np.unique(resi_map)
    residue_seq_list = get_residue_seqIds(resi_list, structure)
    residue_output_array = np.vstack((residue_seq_list, entropy_residues)).T
    pandas_out_residues = pd.DataFrame(residue_output_array)
    pandas_out_residues.columns = ['Residue number', 'Entropy of residue']
    pandas_out_residues.to_csv(output_name)

    print("Saved residue entropies as "+str(output_name))

    return 0 



def main(args):
    fileName = args.matrices
    indices = args.indices
    topology = args.topology


    print("Computing residue entropies now.")
    print("Using the pickle file generated from Enspara")

    print("CARDS file input: "+fileName)

    entropy_vals = extract_entropy_from_matrices(fileName)
    print("Loaded all matrices!")


    resi_map = get_resi_mapping(indices, topology)
    print("Obtained a mapping for each dihedral to its residue")

    residue_level_entropies = sum_entropies_by_residue(resi_map, entropy_vals)

    file_name_without_prepend = fileName[:-7]
    print("Extracted everything! ")

    dihedral_entropy_file = file_name_without_prepend+"_dihedral_entropy.csv"
    save_dihedral_entropy(entropy_vals, resi_map,dihedral_entropy_file)
    
    residue_entropy_file = file_name_without_prepend+"_residue_entropy.csv"
    save_residue_entropy(residue_level_entropies, resi_map, topology, residue_entropy_file)

if __name__ == "__main__":
    #if len(sys.argv) != 7:
    #   print "Usage: 'plot_res_dist.py: This script takes the following arguments: "
    #   print "initialtraj, finaltraj, filetype, structure name, residue, angle \n"
    #Collect Inputs 
    args = parser.parse_args()

    #trajDir = "/home/sukrit/work/vp35"
    #topologyName = "/home/sukrit/work/vp35/prot_only.pdb"
    main(args)



