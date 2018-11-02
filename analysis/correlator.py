#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: sukrit
# @Date:   2015-08-08 15:36:02
# @Last Modified by:   sukrit
# @Last Modified time: 2018-10-26 15:41:30


"""
################### DESCRIPTION #################
#   This is a library of methods used           #
#   to compute the correlated pairs of residues #
#   by doing either structural or disorder      #
#   correlations on either the sidechain or     #
#   backbone of a protein                       # 
#################################################


Notes on Rotameric states
 For ease of calculation, all states are put on a spectrum of 0-360 degrees:
   Any dihedral angle that is negative has 360 added onto it.
   So 181 = -179 degrees and 360 = -1

 Sidechains:
   All sidechains are said to exist in 3 states: 
       0 = gauche- (g-) = +60 = 0-120 degrees = CG between NH and CO of backbone
       1 = trans = 120-240 degrees = CG Next to CO but opposite NH of backbone
       2 = gauche+ (g+) = 240-360 degrees = CG next to NH but opposite CO of backbone

 Backbone
   All phi angles exist in one of two states based on Ramachandran analysis
        0 = state 1 = 0 to 180 degrees
        1 = state 2 = 181 to 360 degrees
    All psi angles exist in one of two states based on Ramachandran analysis
        0 = state 1 =  101 - 260 
        1 = state 2 = 0 - 100 AND 261 to 360 (there's wrap around here)

v2: CHANGELOG: 
    1) Deleted all redundant methods that are not used in the calculation
        a) Holistic state calculations are also included as useless now
    2) Applied comments to all methods for increased clarity 

"""

# Necessary Imports
from . import entropy
import mdtraj as md
import numpy as np
import os
import pylab
import warnings
warnings.filterwarnings("ignore")

################ FUNCTIONS ###################
def get_sidechain_angles(traj):
    """SIDECHAIN METHOD: Extracts the all chi dihedrals, 

    Parameters
    ----------
    traj - Trajectory input file

    Returns
    --------
    angles - list - each element is a 2d array of all the chi angles over time 
    res_names - array - names of each residue for each dihedral
    res_ids - array - sequence id of each dihedral
    atom names - array - The atom of the dihedral being measured
    """
    # Extract Chi Angles and app
    inds, angles = md.compute_chi1(traj)
    n_chis = np.zeros(4) #This array will hold the number of values for each type of chi
    n_chis[0] = inds.shape[0] 
    for i in range(2, 5):
        func = getattr(md, "compute_chi%d" % i)
        more_inds, more_angles = func(traj)
        n_chis[i-1] = more_inds.shape[0]
        inds = np.append(inds, more_inds, axis=0)
        angles = np.append(angles, more_angles, axis=1)
    angles = np.rad2deg(angles)

    # transform so angles range from 0 to 360 instead of -180 to 180
    angles[np.where(angles<0)] += 360

    # get name info
    res_ids = np.array(range(inds.shape[0]), dtype=int)
    #res_names = np.array(range(inds.shape[0]), dtype=str)
    res_names = []
    atom_names = np.array(range(inds.shape[0]), dtype=str)
    for i in range(inds.shape[0]):
        atom = traj.topology.atom(inds[i,2])
        res_ids[i] = int(atom.residue.resSeq)
        #res_names[i] = str(atom.residue.name[:3])
        res_names.append(str(atom.residue.name[:3]))
        atom_names[i] = str(atom.name)


    #Process res_ids to account for multimers 
    num_chains = traj.topology.n_chains
    if (num_chains == 2):
    #If homodimer: add one n_residues value onto the second half of res_ids 
        if (traj.topology.chain(0).n_residues == traj.topology.chain(1).n_residues):
            #Generate an array of all second chain index vals in res_ids
            # The chi values for chain B come right after the corresponding chi values for chain A (they are collated. first chain A then chain B then next chi angle...) 
            sec_chain_indices = np.zeros(4)
            tot_chis = int(sum(n_chis))
            sec_chain_indices[0] = int(n_chis[0]/2)# Chain B chi1 starts here
            sec_chain_indices[1] = int(n_chis[0] + (n_chis[1]/2))
            sec_chain_indices[2] = int(n_chis[0] + n_chis[1] + (n_chis[2]/2))
            sec_chain_indices[3] = int(n_chis[0] + n_chis[1] + n_chis[2]+(n_chis[3]/2))

            res_ids[sec_chain_indices[0]:sec_chain_indices[0]+(n_chis[0]/2)+1] += int(traj.topology.n_residues/2)
            res_ids[sec_chain_indices[1]:sec_chain_indices[1]+(n_chis[1]/2)] += int(traj.topology.n_residues/2)
            res_ids[sec_chain_indices[2]:sec_chain_indices[2]+(n_chis[2]/2)] += int(traj.topology.n_residues/2)
            res_ids[sec_chain_indices[3]:] += int(traj.topology.n_residues/2)

    return angles, res_names, res_ids, atom_names

def get_backbone_angles(traj):
    """BACKBONE METHOD: Extracts the phi and psi dihedrals, 

    Parameters
    ----------
    traj - Trajectory input file

    Returns
    --------
    angles - list - each element is a 2d array of all the phi and psi angles over time 
    res_names - array - names of each residue for each dihedral
    res_ids - array - sequence id of each dihedral
    atom names - array - The atom of the dihedral being measured
    """
    # Extract phi and psi Angles and append them together
    inds, phi_angles = md.compute_phi(traj)
    func = getattr(md, "compute_psi")
    psi_inds, psi_angles = func(traj)
    #inds = np.append(inds, more_inds, axis=0)
    #angles = np.append(angles, more_angles, axis=1)
    
    #convert angles from rad to degrees
    phi = np.rad2deg(phi_angles)
    psi = np.rad2deg(psi_angles)

    #transform so angles range from 0 to 360 instead of -180 to 180
    #Now -180 -> 0 ranges from 181 -> 360 
    phi[np.where(phi<0)] += 360
    psi[np.where(psi<0)] += 360

    # get name info
    # We only need one set of backbone atoms - each residue has 1 phi and 1 psi - BUT when we use only phis or psi inds we'll be missing the first or last angle respectively
    #This is accounted for after the for loop
    res_ids = np.array(range(inds.shape[0]), dtype=int)
    #res_names = np.array(range(inds.shape[0]), dtype=str)
    res_names = []
    atom_names = np.array(range(inds.shape[0]), dtype=str)
    for i in range(inds.shape[0]):
        atom = traj.topology.atom(inds[i,2])
        res_ids[i] = int(atom.residue.resSeq)
        #res_names[i] = str(atom.residue.name[:3])
        res_names.append(str(atom.residue.name[:3]))
        atom_names[i] = str(atom.name)

    #Add the first res_id (left out of phi_inds) by pulling it from psi_inds
    #atom = traj.topology.atom(psi_inds[0,2])
    #firstRes = int(atom.residue.resSeq)


    #Process res_ids to account for multimers 
    num_chains = traj.topology.n_chains
    if (num_chains == 2):
    #If homodimer: add n_residues value onto the second half of res_ids 
        if (traj.topology.chain(0).n_residues == traj.topology.chain(1).n_residues):
            #Halfway down the res_ids shape is the number of residues in the homodimer. Add that onto each res id in the second half.
            res_ids[int(res_ids.shape[0]/2):] += int(traj.topology.n_residues/2)
        
    return phi, psi, res_names, res_ids, atom_names



def assign_angle_to_basin_hard_cutoff(angle):
    """SIDECHAIN METHOD: Assign to states: 0 to 120 -> 0, 120 to 240 -> 1, 240 to 360 -> 2.
    
    Parameters
    ----------
    angle - int 
    
    Returns
    ----------
    state - int - rotameric basin 
    """
    
    if (0 <= angle < 120):
        return 0
    elif (120 <= angle < 240):
        return 1
    elif (240 <= angle <=360):
        return 2
    else:
        print("ERROR: angle", angle, "is not between 0 and 360")
        return np.inf
    
def assign_angles_to_basins_hard_cutoff(angles):
    """ SIDECHAIN METHOD: Converts a set of angles from a trajectory to the basins

    Parameters
    ----------
    angle - int 
    
    Returns
    ----------
    state - int - rotameric basin 


    """
    states = map(assign_angle_to_basin_hard_cutoff, angles)
    new_traj = np.array(states)

    cur_state = new_traj[0]
    transition_times = []
    for i in range(1, new_traj.shape[0]):
        new_state = new_traj[i]
        if new_state != cur_state:
            cur_state = new_state
            transition_times.append(i)

    transition_times = np.array(transition_times)

    return new_traj, transition_times
    

def obtain_transition_times(states):
    """
    This method, given a single trajectory of states over time, identifies all the pionts where a state change occurs

    Parameters
    ----------
    states - 1D array of int - Each element in this array is the state of 
        the dihedral at that point in the trajectory 
    
    Returns
    ----------
    transition-times - 1D array of int - Each element is the array index in
        "states" array where a transition occurred.

    """
    return np.where(states[:-1] != states[1:])[0]


def is_buffered_transition(cur_state, new_angle, buffer_width):
    """ 
    SIDECHAIN METHOD: Decides if a transition in chi angle state counts under the definition of a "buffered" transition. 

    Parameters
    ----------
    cur_state - int - Current state of the sidechain dihedral 
    new_angle - float - the current next angle of the dihedral in the 
        trajectory
    buffer_width - int - size of buffer zone centered around the barrier 
    
    Returns
    ----------
    result - int - returns a 1 or 0 if there is or is not a buffered
        transition occurring, respectively.  
    """
    result = 0
    if cur_state == 0:
        # assume sayed unless crossed into region outside of state
        result = 0
        if (120+0.5*buffer_width < new_angle < 360-0.5*buffer_width):
            result = 1
    elif cur_state == 1:
        # assume moved unless still in region included in state
        result = 1
        if (120-0.5*buffer_width <= new_angle <= 240+0.5*buffer_width):
            result = 0
    elif cur_state == 2:
        # assume sayed unless crossed into region outside of state
        result = 0
        if (0+0.5*buffer_width < new_angle < 240-0.5*buffer_width):
            result = 1

    return result
    
def assign_angles_to_basins_buffered(angles, buffer_width=30):
    """ 
    SIDECHAIN METHOD: Given a trajectory of a single dihedral angle over time, and a buffer width, computes the rotameric states for the angle over the course of the trajectory, where changes in state only occur under the buffered definition of transitions. 

    Parameters
    ----------
    angles - 1D array of float -this is an array of the angles of a single 
        sidechain dihedral over time 
    buffer_width - int - the size of the bufferzone centered around the 
        barriers
    
    Returns
    ----------
    new_traj - 1D array of int - array of all the states of the dihedral 
        over time
    transition_times - 1D array of int - array of indices where a change in
        state occurred  
    """
    new_traj = np.zeros(angles.shape[0])
    transition_times = []
    cur_state = assign_angle_to_basin_hard_cutoff(angles[0])
    new_traj[0] = cur_state
    
    for i in range(1, angles.shape[0]):
        new_angle = angles[i]
        if is_buffered_transition(cur_state, new_angle, buffer_width):
            cur_state = assign_angle_to_basin_hard_cutoff(new_angle)
            transition_times.append(i)
        new_traj[i] = cur_state

    transition_times = np.array(transition_times)

    return new_traj, transition_times

def assign_phi_angle_to_basin_hard_cutoff(angle):
    """BACKBONE METHOD: Assign to states: 0-> 180 = 0; 181 -> 360  = 1
    
    Parameters
    ----------
    angle - int 
    
    Returns
    ----------
    state - int - rotameric basin 
    """
    
    if (0 <= angle < 180):
        return 0
    elif (180 <= angle < 360):
        return 1
    else:
        print("ERROR: angle", angle, "is not between 0 and 360")
        return np.inf
    
def assign_psi_angle_to_basin_hard_cutoff(angle):
    """BACKBONE METHOD: Assign to states: 0-> 100 = 1; 101 -> 260  = 0; 260 -> 360 = 1
    
    Parameters
    ----------
    angle - int 
    
    Returns
    ----------
    state - int - rotameric basin 
    """
    
    if (0 <= angle < 100):
        return 1
    elif (100 <= angle < 260):
        return 0
    elif (260 <= angle <= 360):
        return 1
    else:
        print("ERROR: angle", angle, "is not between 0 and 360")
        return np.inf

def assign_all_phis_hard_cutoff(phi_angles):
    """ BACKBONE METHOD: Assigns a trajectory of a all phi angles over time to its rotameric states 

        Takes all phi angles as a single 2D array and computes the mapping

    Parameters
    ----------
    phi_angles - 2D array of float - matrix containing all phi angles over
        course of the trajectory. Each column is a single phi angle while 
        each row is a separate timestep
  
    """
    states = map(assign_phi_angle_to_basin_hard_cutoff, angles)
    new_traj = np.array(states)

    #Collect all moments of transitions occurring 
    cur_state = new_traj[0]
    transition_times = []
    for i in range(1, new_traj.shape[0]):
        new_state = new_traj[i]
        if new_state != cur_state:
            cur_state = new_state
            transition_times.append(i)

    transition_times = np.array(transition_times)

    return new_traj, transition_times

def assign_all_psis_hard_cutoff(psi_angles):
    """ BACKBONE METHOD: Assigns a trajectory of a single psi angle over time to its rotameric states 

        Takes all psi angles as a single 2D array and computes the mapping
    """
    states = map(assign_psi_angle_to_basin_hard_cutoff, angles)
    new_traj = np.array(states)

    cur_state = new_traj[0]
    transition_times = []
    for i in range(1, new_traj.shape[0]):
        new_state = new_traj[i]
        if new_state != cur_state:
            cur_state = new_state
            transition_times.append(i)

    transition_times = np.array(transition_times)

    return new_traj, transition_times


def assign_backbone_basins_hard_cutoff(phi_angles, psi_angles):
    """ BACKBONE METHOD: Assigns all phi and psi angles to their rotamer states over the course of a trajectory 

    Both phi_angles and psi_angles are given as 2D matrices. Each column 
    is a single individual angle and each row is a single timestep 
    """
    phi_states, phi_transition_times = assign_all_phis_hard_cutoff(phi_angles)
    psi_states, psi_transition_times = assign_all_psis_hard_cutoff(psi_angles)

    return phi_states, phi_transition_times, psi_states, psi_transition_times

def phi_is_buffered_transition(cur_state, new_angle, buffer_width=30):
    """ BACKBONE METHOD: Computes whether or not a transition between phi rotameric states is actually a transition

    NOTE: Barriers exist at 0 and 180 degrees; there are 2 basins to transition between
    """
    result = 0
    if cur_state == 0:
        # assume stayed unless crossed into region outside of state
        result = 0
        if (180+0.5*buffer_width < new_angle < 360-0.5*buffer_width):
            result = 1
    elif cur_state == 1:
        # assume stayed unless crossed into region outside of state
        result = 0
        if (0+0.5*buffer_width < new_angle < 180-0.5*buffer_width):
            result = 1

    return result


def assign_phi_angles_to_basins_buffered(phi_angles, buffer_width=30):
    """ BACKBONE METHOD: Given a set of phi angles, computes the rotameric states of all phi angles over time using the definition of the "buffered transition"
    """
    angles = phi_angles.copy()
    new_traj = np.zeros(angles.shape[0])
    
    transition_times = []
    cur_state = assign_phi_angle_to_basin_hard_cutoff(angles[0])
    new_traj[0] = cur_state
    
    for i in range(1, angles.shape[0]):
        new_angle = angles[i]
        if phi_is_buffered_transition(cur_state, new_angle, buffer_width):
            cur_state = assign_phi_angle_to_basin_hard_cutoff(new_angle)
            transition_times.append(i)
        new_traj[i] = cur_state


    transition_times = np.array(transition_times)

    return new_traj, transition_times

def psi_is_buffered_transition(cur_state, new_angle, buffer_width):
    """BACKBONE METHOD: Computes whether or not a transition between phi angle rotameric states is actually a transition

    NOTE: Barriers exist at 100 and 260 degrees; there are 2 basins to transition between
    """
    result = 0
    if cur_state == 0:
        # assume stayed unless crossed into region outside of state
        result = 0
        if (0 <= new_angle < 100-0.5*buffer_width):
            result = 1
        if (260+0.5*buffer_width < new_angle <= 360 ):
            result = 1
    elif cur_state == 1:
        # assume stayed unless crossed into region outside of state
        result = 0
        if (100+0.5*buffer_width < new_angle < 260-0.5*buffer_width):
            result = 1

    return result


def assign_psi_angles_to_basins_buffered(psi_angles, buffer_width=30):
    """ BACKBONE METHOD: Assigns all phi angles to rotameric states using the "buffered transition" definition

    """
    angles = psi_angles.copy()
    new_traj = np.zeros(angles.shape[0])
    
    transition_times = []
    cur_state = assign_psi_angle_to_basin_hard_cutoff(angles[0])
    new_traj[0] = cur_state
    
    for i in range(1, angles.shape[0]):
        new_angle = angles[i]
        if psi_is_buffered_transition(cur_state, new_angle, buffer_width):
            cur_state = assign_psi_angle_to_basin_hard_cutoff(new_angle)
            transition_times.append(i)
        new_traj[i] = cur_state


    transition_times = np.array(transition_times)

    return new_traj, transition_times



def count_state_occurences(state_traj, n_possible_states):
    """ CORRELATION METHOD: Given a trajectory of states over time, computes how many of each type of state occurred. 

    state_traj is a 1D array of the dihedral state trajectory over time. 
    Given the max value possible in n_possible_states, a 1D array of that
        max size is generated, and filled with the populations of each
        possible state
    """
    occurences = np.zeros(n_possible_states)
    state_inds = []
    for i in range(n_possible_states):
        where_i = np.where(state_traj==i)[0]
        occurences[i] = where_i.shape[0]
        state_inds.append(where_i)

    return occurences, state_inds

def count_joint_state_occurences(state_inds1, state_inds2, n_possible_states):
    """ 
    CORRELATION METHOD: Given a trajectory of states over time, computes a matrix of joint states 

    Returns a 2D matrix of matrices. Each matrix contains addresses: when A is state X, how many times is B in state Y. Each element is the
    population for every possible combination of joint-states

    Parameters 
    --------------
    state_inds1, state_inds2 - the indices where each separate state is found
        in each trajectory 
    n_possible_states - number of possible states for both dihedrals 
    """
    joint_occurences = np.zeros((n_possible_states,n_possible_states))
    for i in range(n_possible_states):
        for j in range(n_possible_states):
            joint_occurences[i,j] = np.intersect1d(state_inds1[i], state_inds2[j]).shape[0]

    return joint_occurences
    
def occurences_to_entropies(occurences):
    """
    Given a set of counts, computes the entropy of that set of counts
    """
    return entropy.calc_entropy(occurences.flatten()/occurences.sum())
    
def entropies_to_mi(joint_s, s1, s2):
    """
    Computes MI between 2 dihedrals using the individual entropies and the joint entropy 

    MI = H(X) + H(Y) - H(X,Y)
    """
    return s1+s2-joint_s

    
def get_msm_traj_fn_list(traj_dir, ext="xtc"):
    """ Generates the list of trajectories in a single trajectory of a given filetype 

    NOTE: ALL TRAJECTORIES MUST BEGIN WITH 'trj'
    """
    traj_fns = []
    i = 0
    fn = os.path.join(traj_dir, "trj%d.%s" % (i, ext))
    while os.path.exists(fn):
        traj_fns.append(fn)
        i += 1
        fn = os.path.join(traj_dir, "trj%d.%s" % (i, ext))
        
    return traj_fns
    

def calc_ord_disord_times(transition_times):
    """ DISORDER CORRELATION METHOD: Given the array of transition times for
    a single trajectory, this method computes the ordered and disordered 
    times that result from that trajectory 

    Parameters
    -------------
    transition times is the array of indices where transitions occurred in
        a single dihedrals trajectory. 

    Returns
    --------
    ord_time - this is the average ordered time as a float
    n_ord - this is the number of times that were collected to compute the 
       ordered time 
    disord_time - the average disordered time of the trajectory as a float
    n_disord - the number of disordered times collected to compute the avg

    """
    num_transitions = transition_times.shape[0]
    
    disord_time = 0.0
    n_disord = 0.0
    ord_time = 0.0
    n_ord = 0.0

    if num_transitions == 1:
        waiting_time = transition_times[0]
        n_ord = waiting_time
        ord_time = waiting_time*(waiting_time+1.0)/2
    elif num_transitions > 1:
        time_between_events = np.diff(transition_times)

        # disordered time is average waiting time between events
        disord_time = time_between_events.mean()

        # ordered time is average waiting time until event from any starting point
        max_waiting_times = [transition_times[0]] + time_between_events.tolist()
        max_waiting_times = np.array(max_waiting_times)
        sum_waiting_times = max_waiting_times*(max_waiting_times+1.0)/2
        ord_time = sum_waiting_times.sum()/max_waiting_times.sum()

        # time between first and last event counts towards calculation of disordered time
        n_disord = transition_times[-1]-transition_times[0]

        # time until last even counts towards ordered time
        n_ord = transition_times[-1]

    return ord_time, n_ord, disord_time, n_disord

def get_all_ord_disord_times(transition_times):

    """
    Given a single set of transition times for a single trajectory, 
    calculates all the ordered and disordered times WITHOUT computing the 
    average ordered and disordered times - this method is meant for being 
    able to observe the distribution of ordered and disordered times
    """
    num_transitions = transition_times.shape[0]
    disord_time = 0.0
    n_disord = 0.0
    ord_time = 0.0
    n_ord = 0.0

    if (num_transitions==1):
        waiting_time = transition_times[0]
        n_ord = waiting_time
        ord_time = waiting_time*(waiting_time+1.0)/2
    elif (num_transitions > 1):
        all_disordered_times = np.diff(transition_times)

        # Compute all ordered times - these are each waiting times 
        # From a single disordered time, we ca compute all ordered times 
        all_ord_times = [[]]*(all_disordered_times.shape[0])
        for i in range(all_disordered_times.shape[0]):
            all_ord_times[i] = compute_orderedTimes_from_disorderedTime(int(all_disordered_times[i]))
        all_ord_times = np.concatenate(all_ord_times, axis=0)

        return all_disordered_times, all_ord_times




def compute_orderedTimes_from_disorderedTime(disord_time):
    if (disord_time <= 1.0):
        return np.asarray([0])
    n_ord = disord_time - 1 
    ord_times = np.zeros(n_ord)

    ord_times[0] = n_ord
    for i in range(1, n_ord):
        ord_times[i] = ord_times[i-1] - 1

    return ord_times



def assign_segments_ordered_disordered(transition_times, traj_len, ord_time, disord_time):
    """ DISORDER CORRELATION METHOD: Given the array of transitions, times,
    trajec. length, and average ordered and disordered times, this method 
    goes through the trajectory and assignes ordered and disordered 
    states to every single timestep. 

    Parameters 
    -----------
    transition_times - the array that is the output of the 
        "get_transition_times" method 
    traj_len - length of the trajectory (honestly it's just the size of
        the "transition_times" array)
    ord_time - average ordered time as a float 
    disord_time - average disordered time as a float

    Returns
    ------------
    new_traj - returns the trajectory of 0's and 1's referring to the 
        ordered and disordered state assignments 
    first_time - first time a transition occurs. Can't be disordered before
    last_time - last time a transition occurs. Can't be disordered after
    """
    num_transitions = transition_times.shape[0]
    
    traj = np.zeros(traj_len)
    traj[:] = np.inf
    first_time = 0
    last_time = 0
    # no ordered/disordered segments if two few transitions or timescales are too similar
    if num_transitions < 2 or ord_time < 3*disord_time:
        #print "No distinction"
        return traj, first_time, last_time
    else:
        #print "assigning"
        first_time = transition_times[0]
        last_time = transition_times[-1]
        for i in range(num_transitions-1):
            seg_start = transition_times[i]
            seg_end = transition_times[i+1]
            time_span = seg_end - seg_start
            likelihood_ratio = ord_time/disord_time * np.exp(-time_span*(1./disord_time - 1./ord_time))
            #print "LR", likelihood_ratio
            if likelihood_ratio >= 3.0: # favors disordered
                traj[seg_start:seg_end] = 1.
            else:
                traj[seg_start:seg_end] = 0.

        return traj, first_time, last_time

def sum_dihedrals_by_resis(mi, resis):
    """
    NOTE: Resi needs to be a 1D array that is equal to the length of one side of the matrix. Maps each dihedral along either length to the residue it belongs to.

    Input
    ---------
    mi - 2d array - 2d array of alldihedral communication
    resis - array - list of what each residue number each element in the row
        or column belongs to
    """
    sorted_res_inds= sort_inds_by_res(resis)
    n_res = len(sorted_res_inds.keys())
#    sorted_keys = np.sort(sorted_res_inds.keys())
#    sorted_keys = np.array(sorted_keys, dtype=int)
    sorted_keys = np.asarray(list(sorted_res_inds.keys()))
    new_res_ids = sorted_keys
    #new_res_names = []
    new_mi = np.zeros((n_res,n_res))
    for i in range(n_res):
        inds_i = sorted_res_inds[sorted_keys[i]]
        #inds_ii = np.ix_(inds_i, inds_i)
        #new_mi[i,i] = mi[inds_ii].sum()
        #new_res_names.append(res_names[sorted_res_inds[sorted_keys[i]][0]])
        for j in range(n_res):
            inds_j = sorted_res_inds[sorted_keys[j]]
            inds_ij = np.ix_(inds_i, inds_j)
            new_mi[i,j] = mi[inds_ij].sum()
            #new_mi[j,i] = new_mi[i,j]

    return new_mi, new_res_ids

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

    return sorted_res_inds


def remove_nans(mi):
    """
    This method removes any nan values that occur in the MI matrix.
        None should exist now because there's never any dividing by zero.

    However this method is kept around in case a new method is developed that
        involves potential division by zero. 
    """
    for i in range(mi.shape[0]):
        for j in range(mi.shape[1]):
            if (np.isnan(mi[i,j])): mi[i,j] = 0.0

    return mi


def remove_diags(mat):
    """ 
    Given a matrix of correlations, this method sets the value of the 
        self-correlating pairs to zero 

    This is a post processing step simply done to remove any diagonals in 
        the MI matrix before further analysis is done. The diagonal of an MI
        matrix is the entropy of that dihedral. 

    """
    newmat = mat.copy()
    for i in range(mat.shape[0]):
        newmat[i,i] = 0.0

    return newmat

def generate_normalized_mi_Per_Resi(mi, allresis, pdbFileName):
    """
    This method puts all the methods together to generate a final MI matrix
    on a residue-residue level for correlations. 

    Parameters
    ----------
    mi - 2d array - this is an all-dihedral communication where each element 
        represents the symmetric uncertainty value between 2 dihedrals
    all_resis - array - list of the residue  number (in sequence space) each 
        element in the "mi" input belongs to
    pdbFileName - string - name of the pdb file of the topology. 

    Returns
    --------
    mi_res - 2d array - communication matrix where each element represents 
        normalized MI value between dihedrals (normalized for the number of 
        dihedrals)
    resis - array - array of the residue numbers in sequence space each 
        row/column belongs to.
    """
    #Clean up the matrix properly - Remove all nan values and the diagonals
    # Nan values occur because some residues have zero entropy - hence dividing by zero
    # The diagonals represent the conformational entropy of a residue, we want to subtract those. 

    noNansMatrix  = remove_nans(mi)
    cleanMatrix = remove_diags(noNansMatrix)

    # Sum up the dihedrals now 
    mi_res, resis = sum_dihedrals_by_resis(mi, allresis)
    #Need to remove the diagonals again - 
    #These diagonals (without the previous values) represent the
    #Intra-Residue communication (how much communication do dihedrals have 
    #within the same residue)
    clean_ResMatrix = remove_diags(mi_res)
    mi_res_norm, resis = normalize_mi_by_numDihedrals(mi_res, allresis, pdbFileName)

    final_MI_Matrix = remove_diags(mi_res_norm)
    return final_MI_Matrix, resis

def process_MI_matrix(matrix):
    """
    Clean up the matrix properly - Remove all nan values and the diagonals
    Nan values occur because some residues have zero entropy - hence dividing by zero
    The diagonals represent the conformational entropy of a residue, we want to subtract those. 
    """

    noNansMatrix = remove_nans(matrix)
    finalMatrix = remove_diags(matrix)

    return finalMatrix

def compute_adjacency_matrix(mat, joint_entropies):
    """
    By dividing each entry in the MI matrix by its joint entropy, we can compute a "distance" metric that can be used to generate graphs of communication. 
    """

    n_dihedrals = mat.shape[0]
    adj_matrix = np.zeros(mat.shape)

    for i in range(n_dihedrals):
        for j in range(n_dihedrals):
            distance = compute_graph_distance(mat[i,j], joint_entropies[i,j])
            adj_matrix[i,j] = distance

    return adj_matrix

def compute_graph_distance(mi_val, joint_entropy):

    dXY = joint_entropy - mi_val
    if (joint_entropy==0.0):
        dXY = 0.0
    distance = dXY/joint_entropy 

    return distance 


def calc_js(p, q):
#### Calculates Jensen-Shannon Divergence  
    n_res = len(p)
    js = np.zeros(n_res)
    for i in range(n_res):
        n_dih = min(len(p[i]), len(q[i]))
        for j in range(n_dih):
            js[i] += entropy.calc_js_divergence(p[i][j], q[i][j])

    return js




def collect_str_states(traj_list, buffered=True, prot_name = "blact_prot_masses.pdb"):
    """ STRUCTURAL CORRELATION METHOD: Given the list of all the trajectory
    files. This method loads them up individually and then computes each of
    their dihedral structural state. 

    The final output is a list (size of n_trajs) where each list element is 
    the states matrix for that trajectory. It also returns the number of 
    total dihedral angles in the protein. 
    """

    n_trajs = len(traj_list)
    all_states = []
    all_ftimes = []
    all_ltimes = []

    for m in range(n_trajs):
        traj = md.load(traj_list[m], top=prot_name)
        #newtraj = traj[::10]
        angles, res_names, res_ids, atom_names = get_sidechain_angles(traj)
        #print "Loading trajectory "+str(m)

        states = np.zeros(angles.shape)
        n_angles = angles.shape[1]
        first_times = np.zeros(n_angles)
        last_times = np.zeros(n_angles)
        ord_times= np.zeros(n_angles)
        disord_times = np.zeros(n_angles)
        for i in range(n_angles):
            # print "Traj", str(m)+ "; dihedral ", str(i)
            if buffered:
                states[:,i], transition_times = assign_angles_to_basins_buffered(angles[:,i])
            else:
                states[:,i], transition_times = assign_angles_to_basins_hard_cutoff(angles[:,i])
        all_states.append(states)

    return all_states, n_angles

def compute_traj_sidechain_struc_states(traj, buffered=True):
    """STRUCTURAL CORRELATION METHOD:  For a single trajectory, computes the 
    structural states of all the dihedrals.  
    """
    chi, res_names, ids, atoms = get_sidechain_angles(traj)
    states = np.zeros(chi.shape)
    n_angles = chi.shape[1]
    for i in range(n_angles):
        # print "Traj", str(m)+ "; dihedral ", str(i)
        if buffered:
            states[:,i], transition_times = assign_angles_to_basins_buffered(chi[:,i])
        else:
            states[:,i], transition_times = assign_angles_to_basins_hard_cutoff(chi[:,i])

    return states, res_names, ids, atoms 



def str_states_to_counts(all_states, n_angles):
    """STRUCTURAL CORRELATION METHOD: Given the list of all structural state
    trajectories, and the number of angles in the protein (which is the just
    the dimensions ona  single state matrix), this method computes the count
    populations for all possible states of every single dihedral. 

    Parameters
    -------------
    all_states - list of structural state matrix trajectories. Each list 
        element corresponds to a single trajectory, and is the matrix of dihedral states over time. 
    n_angles - int of number of total dihedrals in the protein

    Output
    ------------
    counts - array of 1D arrays containing the state populations for each 
        individual dihedral 
    joint_counts - array of 2D arrays containing the joint_state populations 
        for each pair of dihedrals 
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    n_trajs = len(all_states)
    state_inds = [[]]*n_angles
    counts = np.zeros((n_angles, 3))
    joint_counts = np.zeros((n_angles, n_angles, 3, 3))
    for m in range(n_trajs):
        states = all_states[m]
        for i in range(n_angles):
            occurences, inds = count_state_occurences(states[:,i], 3)

            state_inds[i] = inds
            counts[i] += occurences
        for i in range(n_angles):
            for j in range(i+1, n_angles):
                joint_occ = count_joint_state_occurences(state_inds[i], state_inds[j], 3)
                joint_counts[i, j] += joint_occ     

    return counts, joint_counts


def str_entropies(counts, joint_counts, n_angles):
    """STRUCTURAL CORRELATION METHOD: Given the counts, jointcounts, and n_angles values, computes the entropies for each diehdral

    Parameters 
    -----------------
    counts - the counts matrices for each dihedral, output of 
        str_states_to_counts
    joint_counts - joint count matrices for each pair of dihedrals, each
        matrix is a 2D matrix of joint counts

    Output 
    -------------
    entropies - array containing the Entropy of each individual dihedral 
    joint_entropies - 2D array containing the joint entropy of each pair
        of dihedrals  

    """
    entropies = np.zeros(n_angles)
    joint_entropies = np.zeros((n_angles,n_angles))

    for i in range(n_angles):
        entropies[i] = occurences_to_entropies(counts[i])
        joint_entropies[i,i] = entropies[i]
    for i in range(n_angles):
        for j in range(i+1, n_angles):
            joint_entropies[i, j] = occurences_to_entropies(joint_counts[i,j])
            joint_entropies[j,i] = joint_entropies[i,j]

    return entropies, joint_entropies

def str_mi(entropies, joint_entropies, n_angles):
    """ STRUCTURAL CORRELATION METHOD: Computes the MI for every single pair of dihedrals given the entropies and joint entropies

    Parameters
    ------------
    entropies - array containing the entropy of individual dihedrals
    joint_entropies - 2D array containing the joint entropy of each
        individual dihedral pair
    n_angles - int of the number of dihedralsx in the protein

    Output
    --------------
    mi - 2D array containing the MI value for every dihedral pair
    """
    mi = joint_entropies.copy()
    for i in range(n_angles):
        for j in range(i+1, n_angles):
            mi[i,j] = entropies_to_mi(joint_entropies[i,j], entropies[i], entropies[j])
            mi[j,i] = mi[i,j]

    return mi



def get_singleTraj_sidechain_dynStates(traj, buffered=True):
    """
    Computes the sidechain disordered states for a single trajectory
    """
    angles, res_names, res_ids, atom_names = get_sidechain_angles(traj)
    states = np.zeros(angles.shape)
    n_angles = angles.shape[1]
    first_times = np.zeros(n_angles)
    last_times = np.zeros(n_angles)
    ord_times = np.zeros(n_angles)
    disord_times = np.zeros(n_angles)

    for i in range(n_angles):
        states[:, i], transition_times = assign_angles_to_basins_buffered(angles[:, i])
        ord_time, n_ord, disord_time, n_disord = calc_ord_disord_times(transition_times)
        ord_times[i]  = ord_time 
        disord_times[i] = disord_time 
        states[:, i], first_times[i], last_times[i] = assign_segments_ordered_disordered(transition_times, angles.shape[0], ord_time, disord_time)
        states[np.where(states==np.inf)] = 0

    return states, res_ids, first_times, last_times 


def get_singleTraj_ordered_disordered_Times(states):
    """ Given a single trajectory's structural state trajectories, this 
    method computes 2 arrays containing every dihedrals mean 
    ordered and disordered times 

    Parameters 
    -----------------
    states - 2D array containing the structural state trajectories for 
        every individual dihedral (columns are dihedrals, rows is time)

    Output
    --------------
    ord_times - array containing the mean ordered time of each individual dihedral
    disord_times - array with the mean disordered time of each individual dihedral

    """
    # Returns 2 arrays, each array contains every dihedrals individual 
    # Ordered and Disordered times in that Trajectory
    #angles, res_names, res_ids, atom_names = get_sidechain_angles(traj)
    #states = np.zeros(angles.shape)
    n_angles = states.shape[1]
    first_times = np.zeros(n_angles)
    last_times = np.zeros(n_angles)
    ord_times = np.zeros(n_angles)
    disord_times = np.zeros(n_angles)

    for i in range(n_angles):
        transition_times = obtain_transition_times(states[:, i])
        ord_time, n_ord, disord_time, n_disord = calc_ord_disord_times(transition_times)
        ord_times[i]  = ord_time 
        disord_times[i] = disord_time 
        #states[:, i], first_times[i], last_times[i] = assign_segments_ordered_disordered(transition_times, angles.shape[0], ord_time, disord_time)
        #states[np.where(states==np.inf)] = 0

    return ord_times, disord_times

def collect_allTrajs_Ordered_DisorderdTimes(all_states):
    """
    This method computes all the trajectory ordered and disordered times for 
    each individual dihedral 

    Parameters 
    --------------
    all_states - list where each element is an array of the structural states
        trajectory

    Output
    -----------------
    all_ord_times - list where each element corresponds to a trajectory's 
        ordered times array (where each array contains the ordered times of
        each individual dihedral)
    all_disord_times -  list where each element corresponds to a 
        trajectory's disordered times array (where each array contains the 
        disordered times of each individual dihedral)
    """
    # Given all structural state matrices, we can compute the number of 
    # Trajectories we must read
    n_trajs = len(all_states)
    all_ord_times = [[]]*n_trajs
    all_disord_times = [[]]*n_trajs

    for m in range(n_trajs):
        currentTraj = all_states[m]
        ord_times, disord_times = get_singleTraj_ordered_disordered_Times(currentTraj)
        all_ord_times[m] = ord_times
        all_disord_times[m] = disord_times

    return all_ord_times, all_disord_times


def compute_average_ord_Disord_Times(all_ord_times, all_disord_times):
    """
    Computes the average ordered and disordered time for each dihedral across
    the entire trajectory set

    Parameters
    ------------
    all_ord_times - list where each element is a trajectory's ordered times 
        array for each dihedral 
    all_disord_times - list where each element is a trajectory's disordered
        times array for each dihedrals 

    Output 
    -----------
    meanOrdTimes - averaged ordered time for each individual dihedral 
    meanDisordTimes - average disordered time for each individual dihedrals. 
    """
    n_ordered = all_ord_times[0].shape[0]
    n_disordered = all_disord_times[0].shape[0]

    meanOrdTimes = np.zeros(n_ordered)
    meanDisordTimes = np.zeros(n_disordered)

    for i in range(n_ordered):
        vals = [x[i] for x in all_ord_times]
        meanOrdTimes[i] = np.mean(vals)

    for i in range(n_disordered):
        vals = [x[i] for x in all_disord_times]
        meanDisordTimes[i] = np.mean(vals)

    return meanOrdTimes, meanDisordTimes


def assign_AllTrajs_dynStates(all_states, avg_ord_times, avg_disord_times):
    """
    Given all the structural states as well as the average ordered and 
    disordered times for each trajectory, assigns ordered and disordered
    states to each dihedral's trajectory across all trajectories. 

    Parameters 
    --------------
    all_states - list where each element is an array of the structural states
        trajectory
    avg_ord_times - averaged ordered time for each individual dihedral 
    avg_disord_times - average disordered time for each individual dihedrals.

    Output
    ---------------
    all_dynStates - all disordered state trajectories of each dihedral 
        across all simulations

    """
    n_trajs = len(all_states)
    all_dynStates = [[]]*n_trajs


    for m in range(len(all_states)):
        angles = all_states[m]
        n_angles = angles.shape[1]
        dynStates = np.zeros(angles.shape)
        first_times = np.zeros(n_angles)
        last_times = np.zeros(n_angles)
        for i in range(n_angles):
            transition_times = obtain_transition_times(angles[:, i])
            dynStates[:, i], first_times[i], last_times[i] = assign_segments_ordered_disordered(transition_times, angles.shape[0], avg_ord_times[i], avg_disord_times[i])
            # Because assigning segments to ordered and disordered outside 
            # The first and last transitions can result in assigning some
            # To "infinity", we simply assign all those to equal 0
            dynStates[dynStates == np.inf] = 0.

        all_dynStates[m] = dynStates

    return all_dynStates


def collect_allTrajs_DynStates(trajlist, prot_name, buffered=True):
    """
    Given the list of all simulation trajectory files, this method computes 
    the disordered states for all of them and collects them into a single 
    list object. 
    """
    #Given the number of structural state matrices 
    n_trajs = len(trajlist)
    all_ord_times = [[]]*n_trajs
    all_disord_times = [[]]*n_trajs

    stateList, phis, chis = collect_all_dihedral_struc_states(trajlist, buffered, prot_name)
    # Initial pass: Go through all state matrices and collect all ordered 
    # and Disordered times for each dihedral 
    all_ord_times, all_disord_times = collect_allTrajs_Ordered_DisorderdTimes(stateList)


    # Compute the average ordered and disordered times for each separate 
    # Across the entire set of trajectories 
    avg_ord_times, avg_disord_times = compute_average_ord_Disord_Times(all_ord_times, all_disord_times)
    np.savetxt("testPeptide_AverageOrderedTime.dat", avg_ord_times)
    np.savetxt("testPeptide_AverageDisorderedTime.dat", avg_disord_times)

    # Second pass: Go through all trajectories AGAIN and this time assign the states 

    dynStates = assign_AllTrajs_dynStates(stateList, avg_ord_times, avg_disord_times)

    return dynStates, phis, chis

def collect_allAngles(trajlist, prot_name):
    """
    Given the list of trajectories, this method collects all the angles from
    all trajectories and then returns them as a list

    all_angles is a list where each element is a matrix of all individual 
        dihedrals from one trajectory file.  As always, columns are an
        individual angle, and rows are time steps.  
    """
    # Go through all trajectectories in trajlist and collect their angles
    # put them into a single list that contains all the angles for all trajs
    n_trajs = len(trajlist)
    all_angles = [[]]*n_trajs
    #angles, phi_ids, chi_ids = get_angles(trajlist[0])

    for i in range(n_trajs):
        currentTraj = md.load(trajlist[i], top=prot_name)
        angles, phi_ids, chi_ids = get_angles(currentTraj)
        all_angles[i] = angles

    return all_angles, phi_ids, chi_ids


def get_angles(traj):
    """
    From a single trajectroy (mdtraj traj object), this collects the dihedral
    backbone and sidechain angles, concatenates them together and returns 
    them
    """
    phi, psi, names, phi_ids, atoms = get_backbone_angles(traj)
    chi, res_names, chi_ids, atoms = get_sidechain_angles(traj)

    final_angles = np.concatenate((phi, psi, chi), axis=1)

    return final_angles, phi_ids, chi_ids


def convert_singleTraj_Dynstates_to_counts(states, first_times, last_times, n_angles):
    """DEPRECATED METHOD - NEED TO RECODE 
    For a single trajectory's disordered states, this method computes the 
    counts and joint counts 

    Parameters
    -------------
    states - 2d Array containing each dihedral's disordered trajectory. 
        Columns are an individual dihedral 
    first_times, last_times - arrays for each dihedral's first and last 
        transition time points. 
    n_angles - number of dihedral angles

    Output
    -------------
    counts - list of all the counts for each dihedral
    corrcounts - same as counts 
    joint_counts 

    """
    state_inds = []
    counts = np.zeros((n_angles, 2))
    corrcounts = np.zeros((n_angles,n_angles, 2))
    joint_counts = np.zeros((n_angles, n_angles, 2, 2))

    for i in range(n_angles):
        # Self correlating loop
        #print "entropies", i
        occurences, inds = count_state_occurences(states[:,i], 2)
        state_inds.append(inds)
        if first_times[i] == last_times[i]: continue
        counts[i] = occurences
#        entropies[i,i] = occurences_to_entropies(occurences)
#        joint_entropies[i,i] = entropies[i,i]
    
    for i in range(n_angles):
        for j in range(i+1, n_angles):
            #print "joint entropies", i, j
            #identify the biggest region that we will correlate
            #first = max(first_times[i], first_times[j])
            #last = min(last_times[i], last_times[j])

            if first == last: continue
            #occurences_i has 2 elements [#zeroes, #ones]
            #Now we are getting off diagonal elements
            occurences_i, inds_i = count_state_occurences(states[:,i], 2)#,first:last,i], 2)
            occurences_j, inds_j = count_state_occurences(states[:,j],2 ) #first:last,j], 2)
            corrcounts[i,j] += occurences_i
            corrcounts[j,i] += occurences_j
#            entropies[i,j] = occurences_to_entropies(occurences_i)
#            entropies[j,i] = occurences_to_entropies(occurences_j)
            joint_occurences = count_joint_state_occurences(inds_i, inds_j, 2)
            joint_counts[i,j] += joint_occurences
#            joint_entropies[i,j] = occurences_to_entropies(joint_occurences)
#            joint_entropies[j,i] = joint_entropies[i,j]
    return counts, corrcounts, joint_counts




def compute_traj_backbone_states(traj, buffered=True):
    """
    Computes the backbone structural states for a single trajectory
    """

    phi_angles, psi_angles, res_names, res_ids, atom_names = get_backbone_angles(traj)
    n_angles = phi_angles.shape[1]
    first_times = np.zeros(n_angles)
    last_times = np.zeros(n_angles)
    ord_times = np.zeros(n_angles)
    disord_times = np.zeros(n_angles)

    phi_states = np.zeros(phi_angles.shape)
    psi_states = np.zeros(psi_angles.shape)
    phi_dynstates = np.zeros(phi_angles.shape)
    psi_dynstates = np.zeros(psi_angles.shape)
    for i in range(n_angles):
            #print "Traj", str(m)+ "; dihedral ", str(i)
        if buffered:
            phi_states[:,i], phi_transition_times = assign_phi_angles_to_basins_buffered(phi_angles[:,i])
            psi_states[:,i], psi_transition_times = assign_psi_angles_to_basins_buffered(psi_angles[:,i])
        else:
            phi_states[:,i], phi_transition_times = assign_all_phis_hard_cutoff(phi_angles[:,i])
            psi_states[:,i], psi_transition_times = assign_all_psis_hard_cutoff(psi_angles[:,i])

        # Phi classified into dynamic states
        ord_time, n_ord, disord_time, n_disord = calc_ord_disord_times(phi_transition_times)
        ord_times[i] = ord_time
        disord_times[i] = disord_time
        phi_dynstates[:,i], first_times[i], last_times[i] = assign_segments_ordered_disordered(phi_transition_times, phi_angles.shape[0], ord_time, disord_time)
        phi_dynstates[np.where(phi_dynstates==np.inf)] = 0

        # Psi classified into dynamic states
        ord_time, n_ord, disord_time, n_disord = calc_ord_disord_times(psi_transition_times)
        ord_times[i] = ord_time
        disord_times[i] = disord_time
        psi_dynstates[:,i], first_times[i], last_times[i] = assign_segments_ordered_disordered(psi_transition_times, psi_angles.shape[0], ord_time, disord_time)
        psi_dynstates[np.where(psi_dynstates==np.inf)] = 0


    return phi_states, phi_dynstates, psi_states, psi_dynstates     

def collect_backbone_states(trajlist, buffered=True, prot_name="blact_prot_masses.pdb"):
    """
    For a given list of all the trajectories, this method will compute 
    the structural states for all backbone dihedrals. 
    """
    phi_stateList = []
    psi_stateList = []
    phi_dynStateList = []
    psi_dynStateList = []
    n_trajs = len(trajlist)

    for i in range(n_trajs):
        traj = md.load(trajlist[i])#,top = prot_name)
        phi_states, phi_dynstates, psi_states, psi_dynstates =compute_traj_backbone_states(traj, buffered)
        phi_stateList.append(phi_states)
        phi_dynStateList.append(phi_dynstates)
        psi_stateList.append(psi_states)
        psi_dynStateList.append(psi_dynstates)

    return phi_stateList, phi_dynStateList, psi_stateList, psi_dynStateList



def compute_all_struc_states(traj, buffered=True):
    """
    For a single trajectory, computes both backbone and sidechain dihedrals
    and concatenates them into a single matrix. This matrix is the output
    """
    #assign states using already existing methods 
    phi_states, phi_dynstates, psi_states, psi_dynstates = compute_traj_backbone_states(traj, buffered)
    phi, psi, names, phi_ids, atoms = get_backbone_angles(traj) 
    chi_states, ch_names, chi_ids, chi_atoms = compute_traj_sidechain_struc_states(traj, buffered)

    # Combine states along axis 1 into a single matrix
    n_angles = phi_states.shape[1]+psi_states.shape[1]+chi_states.shape[1]
    final = np.concatenate((phi_states, psi_states, chi_states), axis=1)

    # Return backbone and sidechain amino acid residue ids
    return final, phi_ids, chi_ids

def collect_all_dihedral_struc_states(trajlist, buffered=True, prot_name="blact_prot_masses.pdb"):
    """
    Given a list of ALL the trajectory files to load, this method loads
    in each trajectories structural states for ALL dihedrals over time (as 
    individual matrices). 

    The final output is a list where each element is a 2D array of a single 
    dihedral over time (derived from a single trajectory)
    """
    all_states = []
    n_trajs = len(trajlist)
    sampleTraj = md.load(trajlist[0], top=prot_name)
    example_states, phi_ids, chi_ids = compute_all_struc_states(sampleTraj, buffered)

    for i in range(n_trajs):
        traj = md.load(trajlist[i], top=prot_name)
        struc_states, phi_ids, chi_ids = compute_all_struc_states(traj, buffered)
        all_states.append(struc_states)

    return all_states, phi_ids, chi_ids


def count_all_dihedrals_struct_states(strucList):
    """
    Given a list of all structural states across all trajectory, this method
    counts them all up, both individually and pairwise. 

    Parameters
    ------------
    strucList - list where each element is a 2D array of structural states
        over time for each dihedral 

    Output
    --------------
    counts - array where each element is an array containing the structural
        state populations for each individual dihedral 
    joint_counts - 2D array where each element is a 2D array with the joint
        state population of each individual dihedrals (refer to count_joint_
        state_occurrences method for what this means)
    """
    n_trajs = len(strucList)
    n_angles = strucList[0].shape[1]
    state_inds = [[]]*n_angles
    counts = np.zeros((n_angles, 3))
    joint_counts = np.zeros((n_angles, n_angles, 3, 3))

    for m in range(n_trajs):
        states = strucList[m]
        #obtain Phi counts
        for i in range(n_angles):
            occurences, inds = count_state_occurences(states[:,i], 3)
            state_inds[i] = inds
            counts[i] += occurences
        for i in range(n_angles):
            for j in range(i+1, n_angles):
                joint_occ = count_joint_state_occurences(state_inds[i], state_inds[j], 3)
                joint_counts[i, j] += joint_occ

    return counts, joint_counts

def compute_all_dihedral_dynStates(traj, buffered=True):
    """
    For a single dihedral, computes the disordered states for all dihedrals. 
    """
    #obtain all states and methods
    phi_states, phi_dynstates, psi_states, psi_dynstates = compute_traj_backbone_states(traj, buffered)
    phi, psi, names, phi_ids, atoms = get_backbone_angles(traj) 
    chi_dynstates, chi_ids, first_times, last_times = get_singleTraj_sidechain_dynStates(traj, buffered)

    #Combine all states into a single matrix
    n_angles = phi_dynstates.shape[1]+psi_dynstates.shape[1]+chi_dynstates.shape[1]
    final = np.concatenate((phi_dynstates, psi_dynstates, chi_dynstates), axis=1)

    return final, phi_ids, chi_ids, first_times, last_times, n_angles


def collect_all_dihedral_dyn_states(trajlist, buffered=True, prot_name="blact_prot_masses.pdb"):
    """
    For a list of all the trajectories of interest, this method computes the 
    disordered state trajectories  for each indivdual dihedral and collates 
    them into a single list. 
    """
    all_states = []
    all_ftimes = []
    all_ltimes = []
    n_trajs = len(trajlist)

    for i in range(n_trajs):
        traj = md.load(trajlist[i], top=prot_name)
        dyn_states, phi_ids, chi_ids, first_times, last_times, n_angles = compute_all_dihedral_dynStates(traj, buffered)
        all_states.append(dyn_states)
        all_ftimes.append(first_times)
        all_ltimes.append(last_times)

    return all_states, phi_ids, chi_ids, all_ftimes, all_ltimes



def entr_states_to_counts(all_states, n_angles):
    """
    This method computes the disordered state counts and joint counts

    Parameters 
    ------------
    all_states - list where each element is a 2D array with the disordered 
        states over time
    n_angles - number of dihedral angles 

    Output
    ------------
    counts - array where each element has the disordered state counts
        for each dihedral
    joint_counts - array where each element is a 2D array containing the 
        joint counts distribution for disordered states between a pair
    """

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    n_trajs = len(all_states)
    state_inds = [[]]*n_angles
    counts = np.zeros((n_angles, 2))
    joint_counts = np.zeros((n_angles, n_angles, 2, 2))
    for m in range(n_trajs):
        states = all_states[m]
        for i in range(n_angles):
            occurences, inds = count_state_occurences(states[:,i], 2)

            state_inds[i] = inds
            counts[i] += occurences
        for i in range(n_angles):
            for j in range(i+1, n_angles):
                joint_occ = count_joint_state_occurences(state_inds[i], state_inds[j], 2)
                joint_counts[i, j] += joint_occ     

    return counts, joint_counts

def entr_counts_to_entropies(counts, joint_counts, n_angles):
    """
    Computes the entropy and joint entorpy for each individual dihedral 

    Parameters 
    ---------
    counts - array containing the disordered state counts for each individual
        dihedral
    joint_counts - array where each element is a 2D array containing the 
        joint counts for every pair of dihedrals
    n_angles - number of angles in the protein   

    Output 
    -----------
    entropies - array containing the disordered entropy for each individual 
        dihedral
    joint_entropies - array where each element is the joint entropy between
        a pair of dihedrals
    """
    entropies = np.zeros(n_angles)
    joint_entropies = np.zeros((n_angles,n_angles))

    for i in range(n_angles):
        entropies[i] = occurences_to_entropies(counts[i])
        joint_entropies[i,i] = entropies[i]
    for i in range(n_angles):
        for j in range(i+1, n_angles):
            joint_entropies[i, j] = occurences_to_entropies(joint_counts[i,j])
            joint_entropies[j,i] = joint_entropies[i,j]

    return entropies, joint_entropies

def entr_mi(entropies, joint_entropies, n_angles):
    """
    Computes the mutual information for each pair of dihedrals 

    Parameters 
    -----------
    entropies - array where ech element contains the entropy of each dihedral
    joint_entropy - 2D array where each element contains the joint entropy 
        of each pair of dihedrals

    Output 
    -----------
    mi - matrix of MI values between two dihedrals 
    """
    mi = joint_entropies.copy()
    for i in range(n_angles):
        for j in range(i+1, n_angles):
            mi[i,j] = entropies_to_mi(joint_entropies[i,j], entropies[i], entropies[j])
            mi[j,i] = mi[i,j]

    return mi



def mi_normed_numStates(joint_ent, ent1, ent2, minStates):
    """
    Computes a normalized MI between 2 dihedrals by dividing by the minimum number of states possible for either of the two dihedrals. It is given this minimum number from hol_mi_numStatesNormed.  
    """
    mi = ent1 + ent2 - joint_ent
    norm_val = np.log(minStates)
    if (mi == 0.0):
        return 0.0

    norm_mi = float(mi/norm_val)

    return norm_mi



def count_joint_crossState_occurrences(state_inds1, state_inds2, n_possible_states1, n_possible_states2):
    """ CROSS-STATE CORRELATION METHOD: Computs the joint_counts matrix for 
    a pair of dihedrals 

    Returns a 2D matrix of matrices. Each matrix contains addresses: when A is state X, how many times is B in state Y. Each element is the
    population for every possible combination of joint-states

    Parameters 
    --------------
    state_inds1, state_inds2 - the indices where each separate state is found
        in each trajectory 
    n_possible_states1, n_possible_states2 - number of possible states for both dihedrals 1 and 2, respectively


    """
    joint_occurences = np.zeros((n_possible_states1, n_possible_states2))
    for i in range(n_possible_states1):
        for j in range(n_possible_states2):
            joint_occurences[i,j] = np.intersect1d(state_inds1[i], state_inds2[j]).shape[0]

    return joint_occurences


def compute_singleTraj_StrucA_DynB_Counts(strucStates, entrStates):
    """
    Given the structural and disordered states, this computes the counts 
    and joint counts for each individual dihedral. In this type of 
    cross-State correlation, the joint-counts is built where the structure of
    dihedr

    Parameters 
    ----------
    strucStates - Structural state trajectories over time. 
    entrStates - Entropic state trajectories over time 

    Outputs 
    ------------
    strucCounts - Structural state counts for each dihedrals
    dynCounts - disordered state counts for each dihedrals 
    joint_counts - joint structure-disorder counts for every pair of dihedral
        Each row represents the structural state, while each column 
        represents a disordered state

    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    #n_trajs = len(all_states)
    n_angles = strucStates.shape[1]
    allStrucInds = [[]]*n_angles
    allDynInds = [[]]*n_angles
    strucCounts = np.zeros((n_angles, 3))
    dynCounts = np.zeros((n_angles, 2))
    joint_counts = np.zeros((n_angles, n_angles, 3, 2))
    for i in range(n_angles):
        strucOccurences, strucInds = count_state_occurences(strucStates[:,i],3)
        dynOccurences, dynInds = count_state_occurences(entrStates[:,i], 2)
        allStrucInds[i] = strucInds
        allDynInds[i] = dynInds
        strucCounts[i] += strucOccurences
        dynCounts[i] += dynOccurences
        #print("Counts for dihedrals: "+str(i))
    ##print("Finished Counts")
    for i in range(n_angles):
        for j in range(i+1, n_angles):
            joint_occ = count_joint_crossState_occurrences(allStrucInds[i], allDynInds[j], 3, 2)
            print("Computed Joint Counts for dihedrals: "+str(i)+" "+str(j))
            joint_counts[i,j] += joint_occ
            joint_counts[j,i] += joint_occ

    return strucCounts, dynCounts, joint_counts


def compute_allTraj_strucA_EntrB_Counts(allStrucStates, allEntrStates):
    """
    Computs counts, and joint counts for structure and disordered states for ALL trajectories. Collects and collates them all together. 
    """
    n_angles = allStrucStates[0].shape[1]
    n_trajs = len(allStrucStates)
    final_strucCounts = np.zeros((n_angles, 3))
    final_dynCounts = np.zeros((n_angles, 2))
    final_joint_counts = np.zeros((n_angles, n_angles, 3, 2))

    for m in range(n_trajs):
        strucStates = allStrucStates[m]
        entrStates = allEntrStates[m]
        strucCounts, dynCounts, joint_Counts = compute_singleTraj_StrucA_DynB_Counts(strucStates, entrStates)
        final_strucCounts += strucCounts
        final_dynCounts += dynCounts
        final_joint_counts += joint_Counts

    return final_strucCounts, final_dynCounts, final_joint_counts


def compute_singleTraj_DynA_StrucB_Counts(strucStates, entrStates):
    """
    Given the structural and disordered states, this computes the counts 
    and joint counts for each individual dihedral. In this type of 
    cross-State correlation, the joint-counts is built where the structure of
    dihedr

    Parameters 
    ----------
    strucStates - Structural state trajectories over time. 
    entrStates - Entropic state trajectories over time 

    Outputs 
    ------------
    strucCounts - Structural state counts for each dihedrals
    dynCounts - disordered state counts for each dihedrals 
    joint_counts - joint structure-disorder counts for every pair of dihedral
        Each column represents the structural state, while each row 
        represents a disordered state

    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    #n_trajs = len(all_states)
    n_angles = strucStates.shape[1]
    allStrucInds = [[]]*n_angles
    allDynInds = [[]]*n_angles
    strucCounts = np.zeros((n_angles, 3))
    dynCounts = np.zeros((n_angles, 2))
    joint_counts = np.zeros((n_angles, n_angles, 2, 3))
    for i in range(n_angles):
        strucOccurences, strucInds = count_state_occurences(strucStates[:,i], 3)
        dynOccurences, dynInds = count_state_occurences(entrStates[:,i], 2)
        allStrucInds[i] = strucInds
        allDynInds[i] = dynInds
        strucCounts[i] += strucOccurences
        dynCounts[i] += dynOccurences
        print("Counts for dihedrals: "+str(i))
    print("Finished Counts")
    for i in range(n_angles):
        for j in range(i+1, n_angles):
            joint_occ = count_joint_crossState_occurrences(allDynInds[i], allStrucInds[j], 2, 3)
            print("Computed Joint Counts for dihedrals: "+str(i)+" "+str(j))
            joint_counts[i,j] += joint_occ
            joint_counts[j,i] += joint_occ

    return strucCounts, dynCounts, joint_counts


def compute_allTraj_DynA_StrucB_Counts(allStrucStates, allEntrStates):
    """
    Computes counts, and joint counts for structure and disordered states for ALL trajectories. Collects and collates them all together. 
    """
    n_angles = allStrucStates[0].shape[1]
    n_trajs = len(allStrucStates)
    final_strucCounts = np.zeros((n_angles, 3))
    final_dynCounts = np.zeros((n_angles, 2))
    final_joint_counts = np.zeros((n_angles, n_angles, 2, 3))

    for m in range(n_trajs):
        strucStates = allStrucStates[m]
        entrStates = allEntrStates[m]
        strucCounts, dynCounts, joint_Counts = compute_singleTraj_DynA_StrucB_Counts(strucStates, entrStates)
        final_strucCounts += strucCounts
        final_dynCounts += dynCounts
        final_joint_counts += joint_Counts

    return final_strucCounts, final_dynCounts, final_joint_counts



def compute_Struc_and_Entr_entropies(struc_counts, entr_counts):
    """
    Compute the cross-State entropies for the cross-Correlations. This 
    method focuses on computing the entropies of individual structural and
    disordered states. It also inserts the joint entropies into the
    dihedrals. 
    """
    n_angles = struc_counts.shape[0]
    struc_entropies = np.zeros(n_angles)
    entr_entropies = np.zeros(n_angles)
    joint_entropies = np.zeros((n_angles,n_angles))

    #Generate the entropies for the counts 
    for i in range(n_angles):
        struc_entropies[i] = occurences_to_entropies(struc_counts[i])
        entr_entropies[i] = occurences_to_entropies(entr_counts[i])
        joint_entropies[i,i] = struc_entropies[i] + entr_entropies[i]


    return struc_entropies, entr_entropies, joint_entropies


def compute_crossCount_Entropies(joint_counts, joint_entropies):
    """
    Computes the joint entropies from the joint counts for structure-disorder
    joint_count matrices. 
    """
    n_angles = joint_counts.shape[0]

    #Generate the off-diagonal entropies for the counts 
    for i in range(n_angles):
        for j in range(i+1, n_angles):
            joint_entropies[i, j] = occurences_to_entropies(joint_counts[i,j])
            joint_entropies[j,i] = joint_entropies[i,j]

    return joint_entropies

def compute_crossState_MI(struc_entropies, entr_entropies, joint_entropies):
    """
    computes the cross-State MI for structure-disorder or disorder-structure
    """
    mi = np.zeros(joint_entropies.shape)
    n_angles = mi.shape[0]
    for i in range(n_angles):
        for j in range(i+1, n_angles):
            mi[i,j] = mi_to_redundancy(joint_entropies[i,j], struc_entropies[i], entr_entropies[j])
            mi[j,i] = mi[i,j]

    return mi

def compute_crossMI_StrA_EntrB(struc_entropies, entr_entropies, joint_entropies):
    """
    computes the cross-State MI for structure-disorder
    """
    mi = np.zeros(joint_entropies.shape)
    n_angles = mi.shape[0]
    for i in range(n_angles):
        for j in range(i+1, n_angles):
            mi[i,j] = entropies_to_mi(joint_entropies[i,j], struc_entropies[i], entr_entropies[j])
            mi[j,i] = mi[i,j]

    return mi

def compute_crossMI_EntrA_StrB(struc_entropies, entr_entropies, joint_entropies):
    """
    computes the cross-State MI for disorder-structure
    """
    mi = np.zeros(joint_entropies.shape)
    n_angles = mi.shape[0]
    for i in range(n_angles):
        for j in range(i+1, n_angles):
            mi[i,j] = entropies_to_mi(joint_entropies[i,j], entr_entropies[i], struc_entropies[j])
            mi[j,i] = mi[i,j]

    return mi

def compute_struc_counts(strucList):
    """
    Given a list of all structural states across all trajectory, this method
    counts them all up, both individually and pairwise. 

    Parameters
    ------------
    strucList - list where each element is a 2D array of structural states
        over time for each dihedral 

    Output
    --------------
    counts - array where each element is an array containing the structural
        state populations for each individual dihedral 
    joint_counts - 2D array where each element is a 2D array with the joint
        state population of each individual dihedrals (refer to count_joint_
        state_occurrences method for what this means)
    """
    n_trajs = len(strucList)
    n_angles = strucList[0].shape[1]
    state_inds = [[]]*n_angles
    counts = np.zeros((n_angles, 3))
    joint_counts = np.zeros((n_angles, n_angles, 3, 3))

    for m in range(n_trajs):
        states = strucList[m]
        #obtain Phi counts
        for i in range(n_angles):
            occurences, inds = count_state_occurences(states[:,i], 3)
            state_inds[i] = inds
            counts[i] += occurences

    return counts


def compute_dyn_counts(dynList):
    """
    Given a list of all structural states across all trajectory, this method
    counts them all up, both individually and pairwise. 

    Parameters
    ------------
    dynList - list where each element is a 2D array of disordered states
        over time for each dihedral 

    Output
    --------------
    counts - array where each element is an array containing the disordered
        state populations for each individual dihedral 
    joint_counts - 2D array where each element is a 2D array with the joint
        state population of each individual dihedrals (refer to count_joint_
        state_occurrences method for what this means)
    """
    n_trajs = len(dynList)
    n_angles = dynList[0].shape[1]
    state_inds = [[]]*n_angles
    counts = np.zeros((n_angles, 2))
    joint_counts = np.zeros((n_angles, n_angles, 2, 2))

    for m in range(n_trajs):
        states = dynList[m]
        #obtain Phi counts
        for i in range(n_angles):
            occurences, inds = count_state_occurences(states[:,i], 2)
            state_inds[i] = inds
            counts[i] += occurences

    return counts


def compute_singleTraj_StrucCounts(singleTrajStates):
    """
    Given a list of all structural states across one trajectory, this method
    counts them all up, both individually and pairwise. 

    Parameters
    ------------
    singleTrajStates - 2D array where each column is an array of structural 
    states over time for each dihedral 

    Output
    --------------
    counts - array where each element is an array containing the structural
        state populations for each individual dihedral 
    joint_counts - 2D array where each element is a 2D array with the joint
        state population of each individual dihedrals (refer to count_joint_
        state_occurrences method for what this means)
    """
    n_angles = singleTrajStates.shape[1]
    counts = np.zeros((n_angles, 3))
    joint_counts = np.zeros((n_angles, n_angles, 3, 3))
    state_inds = [[]]*n_angles

    for i in range(n_angles):
        occurences, inds = count_state_occurences(singleTrajStates[:,i], 3)
        state_inds[i] = inds
        counts[i] += occurences
    for i in range(n_angles):
        for j in range(i+1, n_angles):
            joint_occ = count_joint_state_occurences(state_inds[i], state_inds[j], 3)
            joint_counts[i, j] += joint_occ

    return counts, joint_counts


def compute_singleTraj_DynCounts(singleTrajStates):
    """
    Given a list of all disordered states across one trajectory, this method
    counts them all up, both individually and pairwise. 

    Parameters
    ------------
    singleTrajStates - 2D array where each column is an array of disordered 
    states over time for each dihedral 

    Output
    --------------
    counts - array where each element is an array containing the disordered
        state populations for each individual dihedral 
    joint_counts - 2D array where each element is a 2D array with the joint
        state population of each individual dihedrals (refer to count_joint_
        state_occurrences method for what this means)
    """

    n_angles = singleTrajStates.shape[1]
    counts = np.zeros((n_angles, 2))
    joint_counts = np.zeros((n_angles, n_angles, 2, 2))
    state_inds = [[]] * n_angles

    for i in range(n_angles):
        occurences, inds = count_state_occurences(singleTrajStates[:,i], 2)
        state_inds[i] = inds
        counts[i] += occurences
    for i in range(n_angles):
        for j in range(i+1, n_angles):
            joint_occ = count_joint_state_occurences(state_inds[i], state_inds[j], 2)
            joint_counts[i, j] += joint_occ

    return counts, joint_counts




