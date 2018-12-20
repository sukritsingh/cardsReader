# -*- coding: utf-8 -*-
# @Author: sukrit
# @Date:   2017-07-13 13:54:56
# @Last Modified by:   Sukrit Singh
# @Last Modified time: 2018-12-12 11:56:09



"""
This script generates the holistic MI matrix using the four Raw MI matrices
of each type. Given the directory all the matrices are found in, as well
as the topology, this script will 1) load them all in, 2) normalize the 
matrices and 3) sum them together to generate the holistic MI matrix.
"""

#################### IMPORTS #################
import mdtraj as md 
import os
import numpy as np
import argparse
import pickle


################# INPUT PARSER #####################
parser = argparse.ArgumentParser(description="""
Calculation of all-dihedral correlations between residues
""")


parser.add_argument('-d','--directory', default=os.getcwd(), help = "Path to pickle file containing CARDS matrices")

################### METHODS ####################

def read_matricies(fileName):
	"""
	This residue reads in the structural and dynamical correlation matrices provided
	"""
	with open(fileName, 'rb') as f: 
		matrices = pickle.load(f)

	ss_mi = matrices['Struc_struc_MI']
	dd_mi = matrices['Disorder_disorder_MI']
	ds_mi = matrices['Disorder_struc_MI']
	sd_mi = matrices['Struc_disorder_MI'] 

	return ss_mi, dd_mi, sd_mi, ds_mi


def main(fileName):
	print("Generating holistic correlation matrices now")
	print("Using the pickle file generated from Enspara")

	print("CARDS file input: "+fileName)

	strucMat, dynMat, sdMat, dsMat = read_matricies(fileName)
	print("Loaded all matrices!")

	file_name_without_prepend = fileName[:-7]

	totalDynMI = dynMat + sdMat + dsMat 
	holMI = strucMat + dynMat + sdMat + dsMat 

	print("Saving matrices now...")
	dynamic_mat_filename = file_name_without_prepend+"_totalDisorder_MI.csv"
	np.savetxt(dynamic_mat_filename, totalDynMI, delimiter=",")
	holistic_mat_filename = file_name_without_prepend+"_holistic_MI.csv"
	np.savetxt(holistic_mat_filename, holMI, delimiter=",")
	print("Saved holsitic matrix as "+holistic_mat_filename)
	print("Saved total disorder-communication matrix as "+dynamic_mat_filename)

if __name__ == "__main__":
	#if len(sys.argv) != 7:
	#	print "Usage: 'plot_res_dist.py: This script takes the following arguments: "
	#	print "initialtraj, finaltraj, filetype, structure name, residue, angle \n"
	#Collect Inputs 
	args = parser.parse_args()
	trajDir = args.directory


	#trajDir = "/home/sukrit/work/vp35"
	#topologyName = "/home/sukrit/work/vp35/prot_only.pdb"
	main(trajDir)




