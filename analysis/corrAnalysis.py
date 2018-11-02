#!/usr/bin/env/python

# AUTHOR: Sukrit Singh
# Developed in the Greg R. Bowman Lab, Washington University in St. Louis
# Version 1
# Last Updated: April 2015



"""
################### DESCRIPTION #################
# This is a library of methods used in analysis #
# of dynamic and structural correlation data. 	#
# To use this data, along with the matricies, 	#
# one also will want a list of residues whose 	#
# dihedrals were analyzed during the correlation#
# analysis. 									#
#################################################
"""

################ IMPORTS ###################
import numpy as np
import os
import pylab
import warnings
import mdtraj as md
#import h5py as h5
import correlator as corr
import math 
from scipy.stats import pearsonr 
from scipy.stats import spearmanr
from scipy.stats import skew
from sklearn.utils import resample
warnings.filterwarnings("ignore")

################ FUNCTIONS ###################

def read_matricies(struc_file, dyn_file):
	"""
	This residue reads in the structural and dynamical correlation matrices provided
	"""
	str_mat = np.loadtxt(struc_file, delimiter=" ")
	dyn_mat = np.loadtxt(dyn_file, delimiter=" ")

	return str_mat, dyn_mat

def remove_diags(mat):
	""" 
	Given a matrix of correlations, this method sets the value of the self-correlating pairs to zero 
	"""
	newmat = mat.copy()
	for i in range(mat.shape[0]):
		newmat[i,i] = 0.0

	return newmat


def filter_matricies(mat, pcentile=90):
	"""
	For a given matrix, finds the minimum value for the desired percentile, and sets anything below that value equal to zero
	"""
	cutoff = np.percentile(mat, pcentile)
	newmat = mat.copy()
	newmat[np.where(newmat<cutoff)] = 0.0

	return newmat

##### Useless method
def connect_pairs_to_correlations(mat, resis):
	xvals, yvals = np.where(mat != 0.0)
	pairs = np.zeros((len(xvals), 3))
	for i in range(len(xvals)):
		pairs[i, 0] = resis[xvals[i]]
		pairs[i, 1] = resis[yvals[i]]
		pairs[i, 2] = mat[xvals[i], yvals[i]]

	return pairs

#### Useless method
def find_residue(resnum, pairs):
	locs = np.where(pairs[:, 0] == resnum)
	resi = pairs[locs]

	return resi


def sumCorrelations_Resis_to_ResiSet(mat, reslist, resis):
	"""
	Given a list of residues A (such as the list of catalytic residues), this method computes the total amount of correlation each residue i has to A 
	"""
	numresis = len(reslist)
	sumList = np.zeros((mat.shape[0]))
	resSum = 0.0
	for i in range(mat.shape[0]):
		row = mat[i, :]
		resSum = 0.0
		for j in range(numresis):
			if (reslist[j] in resis):
				loc = np.where(resis==reslist[j])[0][0]
			else: continue
			resSum += row[loc]
		sumList[i] = resSum

	final = np.zeros((resis.shape[0], 2))
	final[:,0] = resis
	final[:,1] = sumList

	#finalSumList = sumList/(reslist.shape[0]+1)

	return sumList

def normalize_summed_Correlations_to_ResiSet(mat, pdbFile, indices):
	"""
	This method takes the summed correlations for each residue i to set of 
	interesting residues A and normalizes each one by the number of 
	dihedrals being summed over. In this case, we divide 
	"""



### Useless method
def sort_resis_basedOnArray(resis, data):
	yx = zip(data, resis)
	yx.sort()
	resis_sorted = np.flipud([x for y, x in yx])
	data_sorted = np.flipud([y for y, x in yx])

	return resis_sorted, data_sorted


def sumCorrelations_Neighbors(mat, neighbors, resis):
	sumList = np.zeros((mat.shape[0]))
	for i in range(mat.shape[0]):
		resi = mat[i, :]
		nearby = neighbors[i]
		resSum = 0.0
		for j in range(nearby.shape[0]):
			if (nearby[j] in resis):
				loc = np.where(resis==nearby[j])[0][0]
			else: continue
			resSum += resi[loc]
		sumList[i] = resSum

	return sumList

def get_neighbors(pdbfile, resis, cutoff=3.0):
	"""
	Given a distance cutoff in angstroms and a pdb structure, this method computes a list of neighbors R for each residue i where R is all residues within 6 angstroms of the sidechain of i 
	"""
	traj = md.load_pdb(pdbfile)
	allneighbors = [[]] * len(resis)
	for i in range(resis.shape[0]):
		res = resis[i]
		neighbors = get_nearby_resis(traj, res, cutoff)
		allneighbors[i] = neighbors


	return allneighbors

def get_nearby_resis(traj, res, cutoff):
	"""
	Given a single residue from a pdbfile traj, this method computes the set of residues that are within the cutoff distance to the residue 
	"""
	sidechain = traj.topology.select('resSeq '+str(res))
	actual_cutoff = cutoff/10.0
	neighboratoms = md.compute_neighbors(traj, actual_cutoff, sidechain)[0]
	resis = []
	for i in range(neighboratoms.shape[0]):
		resi = traj.topology.atom(neighboratoms[i]).residue
		if (resi.resSeq != res):
			resis.append(resi.resSeq)

	return np.unique(resis)

def get_nearby_dihedrals(traj, res, cutoff):
	"""
	Given a single dihedral from a pdbfile traj, this method computes the set of dihedrals that are within the cutoffdistance 
	"""
	dihedral = traj.topology.select()


def calc_corr_to_ResiSet_and_Neighbors(mat, pdbfile, resis, catlist, cutoff=6.0): 
	"""
	For each residue i, returns the sum of MI(i, A) and MI(i,j) where j is
	the set of all neighboring residues within 6 Angstroms of the sidechain,
	and A is the active site
	"""
	fxnal_MI = sumCorrelations_Resis_to_ResiSet(mat, catlist, resis)
	neighborList = get_neighbors(pdbfile, resis, cutoff)
	neighborMI = sumCorrelations_Neighbors(mat, neighborList, resis)

	final_MI = fxnal_MI + neighborMI

	return final_MI

def calc_mean_correlation_to_resiSet(mat, pdbfile, resis, catlist, cutoff=3.0):
	"""
	For each residue i, returns the sum of MI(i, A) and MI(j,A) where j is
	the set of all neighboring residues within 6 Angstroms of the sidechain,
	and A is the active site. We then normalize it by dividing by the number 
	of dihedrals that are being used to sum things up. 
	"""
	fxnalMI = sumCorrelations_Resis_to_ResiSet(mat, catlist, resis)
	neighborList = get_neighbors(pdbfile, resis, cutoff)
	final_MI = np.zeros(fxnalMI.shape)

	for i in range(len(resis)):
		resiMI = fxnalMI[i]
		neighbors = neighborList[i]
		loclist = []
		for j in range(len(neighbors)):
			loc = np.where(resis==neighbors[j])[0]
			numloc = 0
			if (len(loc) != 0): numloc = loc[0]
			loclist.append(numloc)

		loclist = list(filter((0).__ne__,loclist))
		neighborMI = fxnalMI[loclist]
		total = resiMI + neighborMI.sum()
		#median_dist = np.concatenate((np.asarray([resiMI]),neighborMI))
		#norm_total = total/((neighbors.shape[0])+(2*catlist.shape[0])+1)
		norm_total = total/((neighbors.shape[0]+1)*catlist.shape[0])
		#median_total = np.median(median_dist)
		final_MI[i] = norm_total


	#norm_MI = normalize_summed_neighbor_correlations_by_numDihedrals(final_MI, neighborList, resis, pdbfile)

	return final_MI

def compute_dihedral_based_MI_to_resiSet(data, resiSet, allResis, pdbFile, cutoff=3.0):
	"""
	This uses the dihedral matrix to compute the MI value from any residue i 
	to a set of residues of interest S. For each residue i, we find it's 
	neighbors z within the cutoff, and make that the set A. Then for each 
	residue r in S, we compute the MI between the dihedrals of A to the 
	dihedrals of r. 
	allResis refers to the mapping matrix of the all-dihedral matrix value.  
	"""
	resis = np.unique(allResis)
	neighborList = get_neighbors(pdbFile, resis, cutoff)
	print("Computed all Neighbors")
	final_MI = np.zeros((resis.shape[0]))

	for i in range(resis.shape[0]):
		neighbors = neighborList[i]
		#setA represents residue i and its neighbors 
		print("Computing for Residue "+str(resis[i]))
		setA = np.concatenate([np.asarray([resis[i]]),neighbors])
		MI_val, MI_error = compute_MI_setA_to_setB(data, setA, resiSet, allResis)
		final_MI[i] = MI_val
		

	
	return final_MI

def compute_MI_setA_to_setB(data, setAResis, resiSet, allResis):
	"""
	This function takes in two sets of residues, setB being the residues of 
	interest we are computing MI to, and setA being residue i and it's 
	neighbors. We first find where the MI values that correspond from setA 
	to setB. 
	"""

	#sortedResis = np.sort(allResis)
	#sortedData = sort_MI_vals_by_Resi(data, sortedResis)
	#sortedResis = np.sort(allResis)

	#Collect the relevant indicies for set of residues 

	setA_indices = get_resi_indices(setAResis, allResis)
	resiSet_indices = get_resi_indices(resiSet, allResis)
	n_vals = setA_indices.shape[0] * resiSet_indices.shape[0]

	listOfVals = []
	for i in range(setA_indices.shape[0]):
		for j in range(resiSet_indices.shape[0]):
			val = data[setA_indices[i],resiSet_indices[j]]
			listOfVals.append(val)

	print("Obtained all values for Residue")

	# ListOfVals represents the full set of MI values 
	allVals = np.asarray(listOfVals)
	MI_val = np.mean(allVals)/(resiSet_indices.shape[0])
	#MI_mean = compute_errorbar_MI(allVals, 10)

	return MI_val, MI_error

def compute_dihedral_based_MI_to_resiSet_error(data, resiSet, allResis, pdbFile, cutoff=3.0):
	"""
	This uses the dihedral matrix to compute the MI value from any residue i 
	to a set of residues of interest S. For each residue i, we find it's 
	neighbors z within the cutoff, and make that the set A. Then for each 
	residue r in S, we compute the MI between the dihedrals of A to the 
	dihedrals of r. 
	allResis refers to the mapping matrix of the all-dihedral matrix value.  
	"""
	resis = np.unique(allResis)
	neighborList = get_neighbors(pdbFile, resis, cutoff)
	print("Computed all Neighbors")
	final_MI = np.zeros((resis.shape[0],2))

	for i in range(resis.shape[0]):
		neighbors = neighborList[i]
		#setA represents residue i and its neighbors 
		print("Computing for Residue "+str(resis[i]))
		setA = np.concatenate([np.asarray([resis[i]]),neighbors])
		MI_val, MI_error = compute_MI_setA_to_setB_error(data, setA, resiSet, allResis)
		final_MI[i,0] = MI_val
		final_MI[i,1] = MI_error

	
	return final_MI



def compute_MI_setA_to_setB_error(data, setAResis, resiSet, allResis):
	"""
	This function takes in two sets of residues, setB being the residues of 
	interest we are computing MI to, and setA being residue i and it's 
	neighbors. We first find where the MI values that correspond from setA 
	to setB. 
	"""

	#sortedResis = np.sort(allResis)
	#sortedData = sort_MI_vals_by_Resi(data, sortedResis)
	#sortedResis = np.sort(allResis)

	#Collect the relevant indicies for set of residues 

	setA_indices = get_resi_indices(setAResis, allResis)
	resiSet_indices = get_resi_indices(resiSet, allResis)
	n_vals = setA_indices.shape[0] * resiSet_indices.shape[0]

	listOfVals = []
	for i in range(setA_indices.shape[0]):
		for j in range(resiSet_indices.shape[0]):
			val = data[setA_indices[i],resiSet_indices[j]]
			listOfVals.append(val)

	print("Obtained all values for Residue")

	# ListOfVals represents the full set of MI values 
	allVals = np.asarray(listOfVals)
	MI_val = np.mean(allVals)/(resiSet_indices.shape[0])
	MI_mean, MI_error = compute_errorbar_MI(allVals, 10)

	return MI_val, MI_error


def compute_errorbar_MI(data, bootNo):
	"""
	Given a set of data values (specifically the complete set of MI values
	to a target site) - this function will take all those values, resample
	them a certain number of times, and then compute an error bar for the 
	MI value by resampling the value and computing the std. 
	"""

	newData = []

	for i in range(bootNo):
		dataRS = resample(data)
		newData.append(dataRS)

	return np.mean(newData), np.std(newData[newData!=0.0])

def compute_dihedral_based_fanoFactor_MI_to_resiSet(data, resiSet, allResis, pdbFile, cutoff=3.0):
	"""
	This uses the dihedral matrix to compute the MI value from any residue i 
	to a set of residues of interest S. For each residue i, we find it's 
	neighbors z within the cutoff, and make that the set A. Then for each 
	residue r in S, we compute the MI between the dihedrals of A to the 
	dihedrals of r. 
	allResis refers to the mapping matrix of the all-dihedral matrix value.  
	"""
	resis = np.unique(allResis)
	neighborList = get_neighbors(pdbFile, resis, cutoff)
	print("Computed all Neighbors")
	final_MI = np.zeros((resis.shape[0]))

	for i in range(resis.shape[0]):
		neighbors = neighborList[i]
		#setA represents residue i and its neighbors 
		print("Computing for Residue "+str(resis[i]))
		setA = np.concatenate([np.asarray([resis[i]]),neighbors])
		MI_val, MI_error = compute_fanoFactor_MI_setA_to_setB(data, setA, resiSet, allResis)
		final_MI[i] = MI_error/MI_val
		#final_MI[i,0] = MI_val
		#final_MI[i,1] = MI_error

	
	return final_MI

def compute_fanoFactor_MI_setA_to_setB(data, setAResis, resiSet, allResis):
	"""
	This function takes in two sets of residues, setB being the residues of 
	interest we are computing MI to, and setA being residue i and it's 
	neighbors. We first find where the MI values that correspond from setA 
	to setB. 
	"""

	#sortedResis = np.sort(allResis)
	#sortedData = sort_MI_vals_by_Resi(data, sortedResis)
	#sortedResis = np.sort(allResis)

	#Collect the relevant indicies for set of residues 

	setA_indices = get_resi_indices(setAResis, allResis)
	resiSet_indices = get_resi_indices(resiSet, allResis)
	n_vals = setA_indices.shape[0] * resiSet_indices.shape[0]

	listOfVals = []
	for i in range(setA_indices.shape[0]):
		for j in range(resiSet_indices.shape[0]):
			val = data[setA_indices[i],resiSet_indices[j]]
			listOfVals.append(val)

	print("Obtained all values for Residue")

	# ListOfVals represents the full set of MI values 
	allVals = np.asarray(listOfVals)
	MI_val = np.mean(allVals)/(resiSet_indices.shape[0])
	MI_mean, MI_error = compute_fanoFactor_errorbar_MI(allVals, 10)

	return  MI_mean, MI_error


def compute_fanoFactor_errorbar_MI(data, bootNo):
	"""
	Given a set of data values (specifically the complete set of MI values
	to a target site) - this function will take all those values, resample
	them a certain number of times, and then compute an error bar for the 
	MI value by resampling the value and computing the std. 
	"""

	newData = []

	for i in range(bootNo):
		dataRS = resample(data)
		newData.append(dataRS)

	return np.mean(newData), np.var(newData[newData!=0.0])



def compute_dihedral_based_MAD_MI_to_resiSet(data, resiSet, allResis, pdbFile, cutoff=6.0):
	"""
	This uses the dihedral matrix to compute the MI value from any residue i 
	to a set of residues of interest S. For each residue i, we find it's 
	neighbors z within the cutoff, and make that the set A. Then for each 
	residue r in S, we compute the MI between the dihedrals of A to the 
	dihedrals of r. 
	allResis refers to the mapping matrix of the all-dihedral matrix value.  
	"""
	resis = np.unique(allResis)
	neighborList = get_neighbors(pdbFile, resis, cutoff)
	print("Computed all Neighbors")
	final_MI = np.zeros((resis.shape[0]))

	for i in range(resis.shape[0]):
		neighbors = neighborList[i]
		#setA represents residue i and its neighbors 
		print("Computing for Residue "+str(resis[i]))
		setA = np.concatenate([np.asarray([resis[i]]),neighbors])
		MI_val = compute_MAD_MI_setA_to_setB(data, setA, resiSet, allResis)
		final_MI[i] = MI_val

	
	return final_MI



def compute_MAD_MI_setA_to_setB(data, setAResis, resiSet, allResis):
	"""
	This function takes in two sets of residues, setB being the residues of 
	interest we are computing MI to, and setA being residue i and it's 
	neighbors. We first find where the MI values that correspond from setA 
	to setB. 
	"""

	#sortedResis = np.sort(allResis)
	#sortedData = sort_MI_vals_by_Resi(data, sortedResis)
	#sortedResis = np.sort(allResis)

	#Collect the relevant indicies for set of residues 

	setA_indices = get_resi_indices(setAResis, allResis)
	resiSet_indices = get_resi_indices(resiSet, allResis)
	n_vals = setA_indices.shape[0] * resiSet_indices.shape[0]

	listOfVals = []
	for i in range(setA_indices.shape[0]):
		for j in range(resiSet_indices.shape[0]):
			val = data[setA_indices[i],resiSet_indices[j]]
			listOfVals.append(val)

	print("Obtained all values for Residue")

	# ListOfVals represents the full set of MI values 
	allVals = np.asarray(listOfVals)
	MI_val = calc_median_absolute_deviation(allVals[allVals != 0.0])

	return MI_val	



def compute_dihedral_based_median_MI_to_resiSet(data, resiSet, allResis, pdbFile, cutoff=6.0):
	"""
	This uses the dihedral matrix to compute the MI value from any residue i 
	to a set of residues of interest S. For each residue i, we find it's 
	neighbors z within the cutoff, and make that the set A. Then for each 
	residue r in S, we compute the MI between the dihedrals of A to the 
	dihedrals of r. 
	allResis refers to the mapping matrix of the all-dihedral matrix value.  
	"""
	resis = np.unique(allResis)
	neighborList = get_neighbors(pdbFile, resis, cutoff)
	print("Computed all Neighbors")
	final_MI = np.zeros((resis.shape[0]))

	for i in range(resis.shape[0]):
		neighbors = neighborList[i]
		#setA represents residue i and its neighbors 
		print("Computing for Residue "+str(resis[i]))
		setA = np.concatenate([np.asarray([resis[i]]),neighbors])
		MI_val = compute_median_MI_setA_to_setB(data, setA, resiSet, allResis)
		final_MI[i] = MI_val

	
	return final_MI



def compute_median_MI_setA_to_setB(data, setAResis, resiSet, allResis):
	"""
	This function takes in two sets of residues, setB being the residues of 
	interest we are computing MI to, and setA being residue i and it's 
	neighbors. We first find where the MI values that correspond from setA 
	to setB. 
	"""

	#sortedResis = np.sort(allResis)
	#sortedData = sort_MI_vals_by_Resi(data, sortedResis)
	#sortedResis = np.sort(allResis)

	#Collect the relevant indicies for set of residues 

	setA_indices = get_resi_indices(setAResis, allResis)
	resiSet_indices = get_resi_indices(resiSet, allResis)
	n_vals = setA_indices.shape[0] * resiSet_indices.shape[0]

	listOfVals = []
	for i in range(setA_indices.shape[0]):
		for j in range(resiSet_indices.shape[0]):
			val = data[setA_indices[i],resiSet_indices[j]]
			listOfVals.append(val)

	print("Obtained all values for Residue")

	# ListOfVals represents the full set of MI values 
	allVals = np.asarray(listOfVals)
	MI_val = np.median(allVals[allVals != 0.0])

	return MI_val

def compute_dihedral_based_sum_MI_to_resiSet(data, resiSet, allResis, pdbFile, cutoff=6.0):
	"""
	This uses the dihedral matrix to compute the MI value from any residue i 
	to a set of residues of interest S. For each residue i, we find it's 
	neighbors z within the cutoff, and make that the set A. Then for each 
	residue r in S, we compute the MI between the dihedrals of A to the 
	dihedrals of r. 
	allResis refers to the mapping matrix of the all-dihedral matrix value.  
	"""
	resis = np.unique(allResis)
	neighborList = get_neighbors(pdbFile, resis, cutoff)
	print("Computed all Neighbors")
	final_MI = np.zeros((resis.shape[0]))

	for i in range(resis.shape[0]):
		neighbors = neighborList[i]
		#setA represents residue i and its neighbors 
		print("Computing for Residue "+str(resis[i]))
		setA = np.concatenate([np.asarray([resis[i]]),neighbors])
		MI_val = compute_sum_MI_setA_to_setB(data, setA, resiSet, allResis)
		final_MI[i] = MI_val

	
	return final_MI



def compute_sum_MI_setA_to_setB(data, setAResis, resiSet, allResis):
	"""
	This function takes in two sets of residues, setB being the residues of 
	interest we are computing MI to, and setA being residue i and it's 
	neighbors. We first find where the MI values that correspond from setA 
	to setB. 
	"""

	#sortedResis = np.sort(allResis)
	#sortedData = sort_MI_vals_by_Resi(data, sortedResis)
	#sortedResis = np.sort(allResis)

	#Collect the relevant indicies for set of residues 

	setA_indices = get_resi_indices(setAResis, allResis)
	resiSet_indices = get_resi_indices(resiSet, allResis)
	n_vals = setA_indices.shape[0] * resiSet_indices.shape[0]

	listOfVals = []
	for i in range(setA_indices.shape[0]):
		for j in range(resiSet_indices.shape[0]):
			val = data[setA_indices[i],resiSet_indices[j]]
			listOfVals.append(val)

	print("Obtained all values for Residue")

	# ListOfVals represents the full set of MI values 
	allVals = np.asarray(listOfVals)
	MI_val = np.sum(allVals[allVals != 0.0])



	return MI_val


def compute_dihedral_based_skew_MI_to_resiSet(data, resiSet, allResis, pdbFile, cutoff=6.0):
	"""
	This uses the dihedral matrix to compute the MI value from any residue i 
	to a set of residues of interest S. For each residue i, we find it's 
	neighbors z within the cutoff, and make that the set A. Then for each 
	residue r in S, we compute the MI between the dihedrals of A to the 
	dihedrals of r. 
	allResis refers to the mapping matrix of the all-dihedral matrix value.  
	"""
	resis = np.unique(allResis)
	neighborList = get_neighbors(pdbFile, resis, cutoff)
	print("Computed all Neighbors")
	final_MI = np.zeros((resis.shape[0]))

	for i in range(resis.shape[0]):
		neighbors = neighborList[i]
		#setA represents residue i and its neighbors 
		print("Computing for Residue "+str(resis[i]))
		setA = np.concatenate([np.asarray([resis[i]]),neighbors])
		MI_val = compute_skew_MI_setA_to_setB(data, setA, resiSet, allResis)
		final_MI[i] = MI_val

	
	return final_MI



def compute_skew_MI_setA_to_setB(data, setAResis, resiSet, allResis):
	"""
	This function takes in two sets of residues, setB being the residues of 
	interest we are computing MI to, and setA being residue i and it's 
	neighbors. We first find where the MI values that correspond from setA 
	to setB. 
	"""

	#sortedResis = np.sort(allResis)
	#sortedData = sort_MI_vals_by_Resi(data, sortedResis)
	#sortedResis = np.sort(allResis)

	#Collect the relevant indicies for set of residues 

	setA_indices = get_resi_indices(setAResis, allResis)
	resiSet_indices = get_resi_indices(resiSet, allResis)
	n_vals = setA_indices.shape[0] * resiSet_indices.shape[0]

	listOfVals = []
	for i in range(setA_indices.shape[0]):
		for j in range(resiSet_indices.shape[0]):
			val = data[setA_indices[i],resiSet_indices[j]]
			listOfVals.append(val)

	print("Obtained all values for Residue")

	# ListOfVals represents the full set of MI values 
	allVals = np.asarray(listOfVals)
	MI_val = skew(allVals[:])

	return MI_val

def get_resi_indices(setResis, allResis):
	finalList = []

	for i in range(setResis.shape[0]):
		indices = np.where(allResis == setResis[i])[0]
		finalList.extend(indices)

	return np.asarray(finalList)

def sort_MI_vals_by_Resi(data, sortedResis):

	sortedData = np.zeros(data.shape)
	for i in range(data.shape[0]):
		sortedData[i] = [x for y, x in sorted(zip(sortedResis, data[i]))]

	return sortedData


def sort_inds_by_res(res_ids):
    sorted_res_inds = {}
    for i in range(len(res_ids)):
        res = res_ids[i]
        if res not in sorted_res_inds.keys():
            sorted_res_inds[res] = [i]
        else:
            sorted_res_inds[res].append(i)

    for res in sorted_res_inds.keys():
        sorted_res_inds[res] = np.array(sorted_res_inds[res])

    return sorted_res_ind




def medianCorrelations_Resis_to_ResiSet(mat, reslist, resis):
	"""
	Given a list of residues A (such as the list of catalytic residues), this method computes the total amount of correlation each residue i has to A 
	"""
	numresis = len(reslist)
	medianList = [[]]*mat.shape[0] #np.zeros((mat.shape[0]))
	resSum = 0.0
	for i in range(mat.shape[0]):
		row = mat[i, :]
		orig_indices = resis.argsort()
		ndx = orig_indices[np.searchsorted(resis[orig_indices], np.asarray(reslist))]
		vals = row[ndx]
		medianList.append(vals)

	#final = np.zeros((resis.shape[0], 2))
	#final[:,0] = resis
	#final[:,1] = sumList

	#finalSumList = sumList/(reslist.shape[0]+1)

	return np.asarray(medianList)

def calc_median_correlation_to_resiSet(mat, pdbfile, resis, catlist, cutoff=6.0):
	"""
	For each residue i, returns the sum of MI(i, A) and MI(j,A) where j is
	the set of all neighboring residues within 6 Angstroms of the sidechain,
	and A is the active site. We then normalize it by dividing by the number 
	of dihedrals that are being used to sum things up. 
	"""
	fxnalMI = medianCorrelations_Resis_to_ResiSet(mat, catlist, resis)
	neighborList = get_neighbors(pdbfile, resis, cutoff)
	final_MI = np.zeros(fxnalMI.shape)

	for i in range(len(resis)):
		resiMI = fxnalMI[i]
		neighbors = neighborList[i]
		loclist = []
		for j in range(len(neighbors)):
			loc = np.where(resis==neighbors[j])[0]
			numloc = 0
			if (len(loc) != 0): numloc = loc[0]
			loclist.append(numloc)

		loclist = list(filter((0).__ne__,loclist))
		neighborMI = fxnalMI[loclist]
		#total = resiMI + neighborMI.sum()
		#median_dist = np.concatenate((np.asarray([resiMI]),neighborMI))
		median_dist = np.append(neighborMI, resiMI)
		#norm_total = total/((neighbors.shape[0])+(2*catlist.shape[0])+1)
		#norm_total = total/((neighbors.shape[0]+1)*catlist.shape[0])
		median_total = np.median(median_dist)
		final_MI[i] = median_total

	#norm_MI = normalize_summed_neighbor_correlations_by_numDihedrals(final_MI, neighborList, resis, pdbfile)

	return final_MI


def normalize_summed_neighbor_correlations_by_numDihedrals(MI_list, neighborList, resis, pdbFile):
	"""
	For each total_MI value, divide the total by the total number of 
	dihedrals summed into making the value.  
	"""
	# Each element in the MI list is the total MI of that residue's communication
	# We want to identify the total number of dihedrals to divide each value by  
	numDihedralsPerResi = np.zeros(MI_list.shape)
	prot = md.load(pdbFile)
	for i in range(numDihedralsPerResi.shape[0]):
		neighbors = neighborList[i]
		dihedrals = np.zeros(neighbors.shape)
		for j in range(neighbors.shape[0]):
			resiName = prot.topology.residue(np.where(resis==neighbors[j])[0]).name
			numDihedrals = get_num_dihedrals_by_resi(resiName)
			dihedrals[j] = numDihedrals
		numDihedralsPerResi[i] = np.sum(dihedrals)

	norm_MI = MI_list/numDihedralsPerResi

	return norm_MI


def calc_raw_correlation_to_resiSet(mat, pdbfile, resis, catlist, cutoff=6.0):
	"""
	For each residue i, returns the sum of MI(i, A) and MI(j,A) where j is
	the set of all neighboring residues within 6 Angstroms of the sidechain,
	and A is the active site. This is a raw, unnormalized value. 
	"""
	fxnalMI = sumCorrelations_Resis_to_ResiSet(mat, catlist, resis)
	neighborList = get_neighbors(pdbfile, resis, cutoff)
	final_MI = fxnalMI.copy()

	for i in range(len(resis)):
		resiMI = fxnalMI[i]
		neighbors = neighborList[i]
		loclist = []
		for j in range(len(neighbors)):
			loc = np.where(resis==neighbors[j])[0]
			numloc = 0
			if (len(loc) != 0): numloc = loc[0]
			loclist.append(numloc)

		loclist = list(filter((0).__ne__,loclist))
		neighborMI = fxnalMI[loclist]
		total = resiMI + neighborMI.sum()
		#norm_total = float(total/(neighborMI.shape[0]+1))
		final_MI[i] = total

	return final_MI


def pearsonr_two_matrices(mat1, mat2):
	"""
	Computes the pearson R correlation value between the nonzero values of two matrices
	"""
	total = mat1 + mat2
	inds = np.where(total > 0.0)
	values1 = mat1[inds]
	values2 = mat2[inds]

	return pearsonr(values1, values2)


def spearmanr_two_matrices(mat1, mat2):
	"""
	Computes the spearman R correlation value between the nonzero values of two matrices
	"""
	total = mat1 + mat2
	inds = np.where(total > 0.0)
	values1 = mat1[inds]
	values2 = mat2[inds]

	return spearmanr(values1, values2)


def parse_struc_vs_dyn(filename, num_dihedrals):
	"""
	Given the location of the hdf5 file containing all the joint matrices, this method reads every single joint counts matrix, converts it into it's corresponding structural and dynamics matrices, and computes the structural and dynamic MI from them.
	"""
	#Get full file listing from the hdf5 file
	matrixList = get_JointCount_Matrix_List(filename)

	#Create the matrix files for structural counts
	struc_counts = np.zeros((num_dihedrals, 1, 3))
	struc_joint_counts = np.zeros((num_dihedrals, num_dihedrals, 3, 3))

	#Create the matrix files for entropic counts
	dyn_counts = np.zeros((num_dihedrals, 1, 2))
	dyn_joint_counts = np.zeros((num_dihedrals, num_dihedrals, 2,2))

	for i in range(matrixList.shape[0]):
		hol_mat = read_matrix_from_h5(filename, matrixList[i])
		jc_struc, jc_dyn = holistic_to_jointcounts(hol_mat)
		dihedral1, dihedral2 = get_dihedrals_from_JCMatrix_name(matrixList[i])
		#Parse all the structural counts and put them in their appropriate place
		struc1, struc2 = parse_single_jc_matrix(jc_struc, 3)
		#Based on how the joint_counts matrices were saved, we don't need to worry about double counting into each state - there is only one joint counts matrix for a single dihedral
		struc_counts[dihedral1] += struc1
		struc_counts[dihedral2] += struc2
		struc_joint_counts[dihedral1, dihedral2] += jc_struc

		#Parse all the entropic counts and put them in their appropriate place
		dyn1, dyn2 = parse_single_jc_matrix(jc_dyn, 2)
		dyn_counts[dihedral1] += dyn1
		dyn_counts[dihedral2] += dyn2
		dyn_joint_counts[dihedral1, dihedral2] += jc_dyn

	return struc_count, struc_joint_counts, dyn_counts, dyn_joint_counts


def parse_single_jc_matrix(matrix):
	"""
	This matrix takes a single joint count matrix of structural or entropic 
	states and sums them by row and column. Summing by row leads to the 
	first dihedrals counts. Summing by column gives the second dihedral's
	counts. The output is based on the given number of possible states. 
	"""

	output_rows = np.sum(matrix, axis=1) # axis 1 is the rows 
	output_cols = np.sum(matrix, axis=0) # axis 0 is the columns

	return output_rows, output_cols



def get_dihedrals_from_JCMatrix_name(MatrixName):
	"""
	Uses the partition() method for strings to parse out the two dihedrals
	that a matrix is referring two. The first number in the name is the
	rows, the second number is the columns. 
	"""
	#Define partition point substrings
	firstsubstring = '_Joint_Counts'
	#The second substring will split the numbers into separate values
	secondsubstring = '_' 
	
	#Now the actual parititioning
	dihedral_nums = MatrixName.partition(firstsubstring)[0]
	#Get the ints out 
	d1 = int(dihedral_nums.partition(secondsubstring)[0])
	d2 = int(dihedral_nums.paritition(secondsubstring)[2])

	return d1, d2



def get_JointCount_Matrix_List(filename):
	with h5.File(filename, 'r') as hf:
		data = h5.items()
		datalist = list(data)

	data_array = np.asarray(datalist)
	filenames = data_array[:,0]

	return filenames


def read_matrix_from_h5(filename, matrixname):
	"""
	Given the file and name of the matrix, this method extracts the matrix of the given name from the data file and returns it 
	"""
	with h5.File(filename, 'r') as hf:
		data = hf[matrixname][:]

	return data



def holistic_to_jointcounts(hol_Matrix):
	"""
	Given a matrix of the holistic state counts for a single dihedral pair, this method parses the single matrix into the structural and entropic matrix, and then generates the structural and entropic joint count matricies (which can be converted into single count matrices) 
	"""
	# The structural MI matrix is a 3x3 matrix (3 structural states)
	# You add each consecutive pair of Columns together to get the colummn 
	# in ther 3x3 (first and second column are State 1, 3rd and 4th are 
	#State 2...)
	groups = [[0,1],[2,3],[4,5]]
	struc =  np.zeros((3,3))
	for i in range(3):
		vals = []
		pair = groups[i]
		intermediate = hol_Matrix[pair[0]] + hol_Matrix[pair[1]]
		for j in range(0,6,2):
			val = intermediate[j] + intermediate[j+1]
			vals.append(val)
		struc[i] = np.asarray(vals)

	#For entropic MI matrix, we need a final 2x2 matrix (2 states)
	# You add each alternating row and column together to get the
	# corresponding column in the final matrix
	groups = [[0,2,4],[1,3,5]]
	dyn = np.zeros((2,2))
	for i in range(2):
		vals = []
		triplet = groups[i]
		intermediate = hol_Matrix[triplet[0]]+hol_Matrix[triplet[1]]+hol_Matrix[triplet[2]]
		print(intermediate)
		for j in range(2):
			vals_sum = intermediate[j]+intermediate[j+2]+intermediate[j+4]
			vals.append(vals_sum)
		dyn[i] = np.asarray(vals)

	return struc, dyn


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols))

def collect_allDihedral_atoms(pdbFile):
	"""
	This method, given a pdbFile, loads all the relevant atom indicies 
	relevant to each dihedral. Each set of 4 indicies represents one dihedral
	and maps directly to each row or column element in the MI matrix. 
	"""
	# Obtain both sidechain and backbone atoms and concatenate them into a single list 
	backbone_inds = collect_bkbone_atoms_inds(pdbFile)
	sc_inds = collect_sidechain_atom_inds(pdbFile)
	final_inds = np.concatenate((backbone_inds, sc_inds))

	return final_inds


def collect_bkbone_atoms_inds(pdbFile):
    inds, phi_angles = md.compute_phi(pdbFile)
    func = getattr(md, "compute_psi")
    psi_inds, psi_angles = func(pdbFile)
    final_inds = np.concatenate((inds, psi_inds))

    return final_inds

def collect_sidechain_atom_inds(pdbFile):
	# Extract Chi Angles and app
	inds, angles = md.compute_chi1(pdbFile)
	n_chis = np.zeros(4) #This array will hold the number of values for each type of chi
	n_chis[0] = inds.shape[0] 
	for i in range(2, 5):
		func = getattr(md, "compute_chi%d" % i)
		more_inds, more_angles = func(pdbFile)
		n_chis[i-1] = more_inds.shape[0]
		inds = np.append(inds, more_inds, axis=0)
		angles = np.append(angles, more_angles, axis=1)
	angles = np.rad2deg(angles)

	return inds



def calculate_distances_between_AllDihedral_atoms(allInds, pdbFile):
	# Setup a matrix that is the size of allInds on both sides
	dist_matrix = np.zeros((allInds.shape[0], allInds.shape[0]))

	# Start going through each row and column in the distance matrix and populate it
	for i in range(dist_matrix.shape[0]):
		# Do only the area above the diagonal of the matrix, the area below is symmetric 
		for j in range(dist_matrix.shape[0]):
			inds1 = allInds[i]
			inds2 = allInds[j]
			dist_matrix[i,j] = calculate_distance_between_inds(inds1, inds2, pdbFile)
			

	return dist_matrix


def calculate_distance_between_inds(inds1, inds2, pdbFile):

	#For each set of atoms in inds1 and inds2, compute every possible unique pairing
	allPairs  = cartesian([inds1, inds2])
	numPairs = allPairs

	# Compute all the distances between all the pairs and compute the average
	dists = md.compute_distances(pdbFile, allPairs)[0] #MDTraj gives output in nm
	avgDist = np.mean(dists)

	# The value is returned in angstroms
	return avgDist*10 



def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]

    return out

def set_intraResidue_to_zero(row, currentRes, resiMap):
	"""
	This method, given the row of dihedrals, and the residue to which that dihedral belongs to, finds the areas in the row corresponding to dihedrals of the same residue, and sets them to zero. This eliminates us capturing any intra-residue communication
	"""
	locs = np.where(resiMap==currentRes)[0]
	row[locs] = 0.0

	return row


def identify_hotspots(matrix, resiMap):
	"""
	This residue, given a matrix of MI communication between all dihedrals, seeks to find hotspots of communication by summing each row in the matrix. 
	Each summation would represent communication from dihedral i to the whole protein. We will ignore values between dihedrals in the same protein so that we are ignoring any intra-residue communication
	For a single residue, sum up all these dihedral summations. 
	"""
	# Generate the final output list 
	n_dihedrals = matrix.shape[0]
	resis = np.unique(resiMap)
	n_resis = resis.shape[0]
	final_data = np.zeros((n_resis, 2))
	final_data[:,0]  = resis

	# Go through ech row of the matrix 
	for i in range(n_dihedrals):
		row = matrix[i]
		new_row = set_intraResidue_to_zero(row, resiMap[i], resiMap)
		currentRes = resiMap[i]
		loc = np.where(resis == currentRes)[0][0]
		final_data[loc,1] += np.sum(new_row)

	return final_data


def identify_hotspots_include_neighbors(matrix, resiMap, pdbFile, cutoff=3.0):
	"""
	This method does  the same as the identify_hotspots method except for each residue, we also are including the dihedrals of neighboring residues, and we are excluding communication between neighboring residue
	"""
	# Generate list of each residue's neighbors 
	resis = np.unique(resiMap)
	neighborList = get_neighbors(pdbFile, resis, cutoff)

	# Generate final output list 
	n_dihedrals = matrix.shape[0]
	resis = np.unique(resiMap)
	n_resis = resis.shape[0]
	final_data = np.zeros((n_resis, 2))
	final_data[:,0]  = resis

	for i in range(n_resis):
		currentRes = resis[i]
		currentNeighbors = neighborList[i]
		setA = np.concatenate([np.asarray([currentRes]),currentNeighbors])
		hotspotVal = compute_totalMI_include_Neighbors(matrix, setA, resiMap)
		locRes = np.where(resis == currentRes)[0][0]
		final_data[locRes,1] += hotspotVal

	return final_data



def compute_totalMI_include_Neighbors(matrix, resiSet, resiMap):
	"""
	This method takes in the MI matrix, and given an array of the residue and it's neighbors, sums up each of the relevant rowsto give a final value. IntraResidue and Neighbor-neighbor communications are ignored here
	"""
	locs = find_ndexes(resiMap, resiSet)
	totalMI = 0.0

	for i in range(locs.shape[0]):
		row = matrix[locs[i]]
		new_row = set_intraResidue_interNeighbor_MI_to_zero(row, resiSet, resiMap)
		totalMI += np.sum(new_row)

	return totalMI

def identify_hotspots_bootstrapped_wneighbors(matrix, resiMap, pdbFile, cutoff=3.0):
	"""
	This method does  the same as the identify_hotspots method except for each residue, we also are including the dihedrals of neighboring residues, and we are excluding communication between neighboring residue
	"""
	# Generate list of each residue's neighbors 
	resis = np.unique(resiMap)
	neighborList = get_neighbors(pdbFile, resis, cutoff)

	# Generate final output list 
	n_dihedrals = matrix.shape[0]
	resis = np.unique(resiMap)
	n_resis = resis.shape[0]
	final_data = np.zeros((n_resis, 3))
	final_data[:,0]  = resis

	for i in range(n_resis):
		currentRes = resis[i]
		currentNeighbors = neighborList[i]
		setA = np.concatenate([np.asarray([currentRes]),currentNeighbors])
		hotspotVal, hotspotError = compute_totalMI_bootstrapped_wNeighbors(matrix, setA, resiMap)
		locRes = np.where(resis == currentRes)[0][0]
		final_data[locRes,1] += hotspotVal
		final_data[locRes, 2] += hotspotError

	return final_data


def compute_totalMI_bootstrapped_wNeighbors(matrix, resiSet, resiMap):
	"""
	This method takes in the MI matrix, and given an array of the residue and it's neighbors, sums up each of the relevant rowsto give a final value. IntraResidue and Neighbor-neighbor communications are ignored here
	"""
	locs = find_ndexes(resiMap, resiSet)
	totalMI = 0.0
	totalMIError = 0.0

	for i in range(locs.shape[0]):
		row = matrix[locs[i]]
		new_row = set_intraResidue_interNeighbor_MI_to_zero(row, resiSet, resiMap)
		##totalMI += np.sum(new_row)
		newMI, newMIError = bootstrap_globalCorr_singleResidueSet(new_row, 10)
		totalMI += newMI
		totalMIError += newMIError
		#totalMIError += bootstrap_globalCorr_singleResidueSet(new_row, 10)

	return totalMI, totalMIError

def bootstrap_globalCorr_singleResidueSet(rowVals, numberSamples):
	"""Given a row of values, this method outputs a bootstrapped error 
	of the row for further analysis"""

	newData = []
	for i in range(numberSamples):
		dataRS = resample(rowVals)
		newData.append(dataRS)
	newData = np.mean(np.asarray(newData), axis=0)

	##newData = np.asarray(newData)
	return np.sum(newData), np.std(newData)


	#return np.sum(newData), np.std(newData)


def find_ndexes(vals, wantedVals):
	"""
	Given an array and multiple unique values of interest, this method computes the index values for all occurrences of each unique value
	"""
	ix = np.in1d(vals.ravel(), wantedVals).reshape(vals.shape)
	locs = np.where(ix)[0]

	return locs

def find_max_MI_by_Distance(dist_matrix, mi_matrix, bin_width):
	"""
	Given the matrix of interdihedral distances, and the matrix of MI values
	between each dihedral, this matrix computes the maximum MI value at each 
	distance interval. The distance interval is based on the bin width
	"""
	max_Distance = dist_matrix.max()
	n_bins = int(dist_matrix.max()/bin_width)

	maxVals = np.zeros(n_bins)
	# For now: the xvals used will be the maximum distances in each bin
	xvals = np.zeros(n_bins)

	for i in range(n_bins):
		minDist = i*bin_width
		maxdist = minDist + bin_width
		vals = mi_matrix[np.logical_and(dist_matrix > minDist, dist_matrix < maxdist)]

		maxVals[i] = vals.max()
		xvals[i] = maxdist

	return maxVals, xvals


def find_fraction_kT_CorrelatedDihedrals(dist_matrix, mi_matrix, bin_width):
	"""
	Given the matrix of interdihedral distances, and the matrix of MI values
	between each dihedral, this matrix the fraction of dihedrals that have 
	correlation > 1/e for each distance interval. 
	The distance interval is based on the bin width
	"""
	max_Distance = dist_matrix.max()
	n_bins = int(dist_matrix.max()/bin_width)

	maxVals = np.zeros(n_bins)
	# For now: the xvals used will be the maximum distances in each bin
	xvals = np.zeros(n_bins)

	for i in range(n_bins):
		minDist = i*bin_width
		maxdist = minDist + bin_width
		vals = mi_matrix[np.logical_and(dist_matrix > minDist, dist_matrix < maxdist)]
		fracVals = vals[vals > 0.005].shape[0] / vals.shape[0]
		maxVals[i] = fracVals
		xvals[i] = maxdist

	return maxVals, xvals






def set_intraResidue_interNeighbor_MI_to_zero(row, resiSet, resiMap):
	"""
	This method takes in a single row of the MI matrix and sets all the
	locations where resiMap == resiSet to 0.0 so that they don't get summed 
	in. This allows summing the row to only capture total MI while ignoring 
	intraResidue communication and inter-Neighbor communiation. 
	"""
	locs = find_ndexes(resiMap, resiSet)
	row[locs] = 0.0

	return row


def normalize_StrucMI_Matrix(strucMatrix, numBkbone):
	n_backbones = numBkbone #This is how many backbone dihedrals are in CAP 
	#TO DO: Implement so backbone dihedrals are automaticall figured out
	final = np.zeros(strucMatrix.shape)

	final[:n_backbones, :n_backbones] = strucMatrix[:n_backbones, :n_backbones]/(np.log(2))
	final[n_backbones:, :n_backbones] = strucMatrix[n_backbones:, :n_backbones]/(np.log(2))
	final[:n_backbones, n_backbones:] = strucMatrix[:n_backbones,n_backbones:]/(np.log(2))
	final[n_backbones:, n_backbones:] = strucMatrix[n_backbones:, n_backbones:]/(np.log(3))

	return final


def normalize_DynMI_Matrix(dyn_matrix):
	return dyn_matrix/np.log(2)

def normalize_mi_by_numDihedrals(mi, resiList, pdbFileName):
	"""
	Given a list of residues for the protein and the pdb file, this function 
	returns a normalized MI array for residue-residue MI. It normalizes by 
	dividing each MI value by the total number of dihedrals that exist 
	between the two residues, giving an average MI value (of sorts). 

	Before anybody asks: the MI between each residue does NOT incorporate 
	neighbor effects natively. This is ONLY a measure of the direct 
	communication between residues. 
	"""

	#Load the data files used for normalizing
	prot = md.load(pdbFileName)
	resis = np.loadtxt(resiList)
	data = np.loadtxt(mi)
	norm_data = np.zeros(data.shape)

	#Start a for loop to go through each element individually
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			xres = prot.topology.residue(i).name
			yres = prot.topology.residue(j).name
			numXDihedrals = get_num_dihedrals_by_resi(xres)
			numYDihedrals = get_num_dihedrals_by_resi(yres)
			totalDihedrals = numXDihedrals+numYDihedrals
			norm_data[i,j] = float(data[i,j]/totalDihedrals)

	return norm_data


def convert_MI_Matrix_To_Distances(miMat, jointEntropyMat):
	"""
	Given the matrix of mutual information between dihedrals(or whatever)
	and the joint Entropy of each dihedral, this matrix computes the distance
	(AKA Similarity Metric) of that dihedral pair. 
	"""
	ratio = miMat/jointEntropyMat

	return (1-ratio)
	


def get_num_dihedrals_by_resi(resiType):
	"""
	This returns the number of dihedrals in a given residue type. 
	"""
	if (resiType=="ARG" or resiType == "LYS"):
		return 6
	elif (resiType=="MET" or resiType=="GLU" or resiType=="GLN"):
		return 5
	elif (resiType=="TYR" or resiType=="TRP" or resiType=="PRO" or resiType=="PHE" or resiType=="LEU" or resiType=="ILE" or resiType=="HIS" or resiType=="ASP" or resiType=="ASN"):
		return 4
	else:
		return 3


def calc_median_absolute_deviation(mi_vals):
	median = np.median(mi_vals)
	med_deviations = np.abs(mi_vals - median)

	return np.median(med_deviations)

def calc_MAD_based_std_score(mi_vals):
	mad_val = calc_median_absolute_deviation(mi_vals)
	median = np.median(mi_vals)
	med_deviations = np.abs(mi_vals - median)
	score = med_deviations/mad_val

	return score

def plot_angles(angles, buffer_width=30):
    """ 
    Plots a single dihedral's trajectory over time
    """
    n = angles.shape[0]
    d = 0.5*buffer_width
    pylab.plot(angles)
    pylab.plot([0, n], [0, 0], 'k', linewidth=2)
    pylab.plot([0, n], [120, 120], 'k', linewidth=2)
    pylab.plot([0, n], [240, 240], 'k', linewidth=2)
    pylab.plot([0, n], [360, 360], 'k', linewidth=2)
    pylab.plot([0, n], [d, d], 'm', linewidth=1)
    pylab.plot([0, n], [120-d, 120-d], 'm', linewidth=1)
    pylab.plot([0, n], [120+d, 120+d], 'm', linewidth=1)
    pylab.plot([0, n], [240-d, 240-d], 'm', linewidth=1)
    pylab.plot([0, n], [240+d, 240+d], 'm', linewidth=1)
    pylab.plot([0, n], [360-d, 360-d], 'm', linewidth=1)

def print_for_pymol(res_ids):
    s = "%d" % res_ids[0]
    for r in res_ids[1:]:
        s += "+%d" % r
    print(s)