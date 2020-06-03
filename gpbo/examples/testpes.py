from util import *
import gpbo
import scipy as sp
import numpy as np
import os
from datetime import datetime
import sys
from PES.main import *

"""
This script shows how to:
    1) Call functions to construct the Ising spin glass hamiltonian
    from the optimization problem stored in mps files
    2) call the expectation value function for iteration level p=1 (2D function)
    3) compute the final qaoa state of those angles
    4) visualize the energylandscape for 2D

How to execute QAOA:
1) find optimal gamma beta angles for 
   p = 1, 2 ...., N where N is chosen by the user

   ! this is where you should use BLOSSOM :) 

2) compute the final qaoa states. 
   the probability of obtaining correct result is 
   np.abs(final state[optimal_index])^2 
   where the qaoa state is the quantum state stored in a column vector

   ! You should save and plot the following:
            1) optimal angles for each level of p 
            2) success probability of each level of p
            3) save the final qaoa states and greate histogram

            plot these items 
"""

# read data
# constraint matrix
file_location = 'data_instances/' 
file_name = 'instance_8_0.mps'


[c, A, nbr_of_flights, nbr_of_routes] = read_data_from_mps(file_location, file_name)

nbr_of_qubits = nbr_of_routes
nbr_states = 2**nbr_of_qubits



# compute the ising spin glass hamiltonian
[linear_cost_hamiltonian, 
quadratic_cost_hamiltonian, 
cost_hamiltonian] =  create_ising_hamiltonian(c, 
                             A, 
                             nbr_of_qubits, 
                             nbr_of_flights)

# Exact_cover:
H = quadratic_cost_hamiltonian

# mixer operator 
sigmax = np.array([[0, 1], 
                   [1, 0]])

initial_state = 1.0/np.sqrt(nbr_states)*np.ones((nbr_states, 1))

# compute the state for a variational angle for a certain level of p
# the dimension of the expectation value function is 2*p
p = 1

def fblossom(beta_gamma_angles,**ev):
    beta_gamma_angles[:p] = [b*2*np.pi for b in beta_gamma_angles[:p]]
    beta_gamma_angles[p:2*p] = [b*np.pi for b in beta_gamma_angles[p:2*p]]
    c = 1.

    E = expectation_value(beta_gamma_angles, H, p, nbr_of_qubits, initial_state,sigmax)
    return E, c, dict()

def f(beta_gamma_angles,**ev):
    beta_gamma_angles[:p] = [b*2*np.pi for b in beta_gamma_angles[:p]]
    beta_gamma_angles[p:2*p] = [b*np.pi for b in beta_gamma_angles[p:2*p]]
    c = 1.

    E = expectation_value(beta_gamma_angles, H, p, nbr_of_qubits, initial_state,sigmax)
    return E
    
#beta = np.pi/2
#gamma = np.pi/3
#beta_gamma_angles = np.array([gamma, beta])
#print(beta_gamma_angles)

#noise
#s = 0

#C = gpbo.core.config.switchdefault(f, p*2, 10, 500*(p+1), s, 'results', 'qaoa-test.csv')
#C.choosepara['regretswitch'] = 1e-2
#C.choosepara['pvetol'] = 1e-2
#C.aqpara[1]['tol']=None

#out = gpbo.search(C, False)
#print(out)
#print('True value for noisy functions')
#f(out[0],cheattrue=True)[0]

#beta_gamma_angles = out[0]


#Specify the parameters for running PES
target_function = f
x_minimum = np.asarray([0.0,0.0])
x_maximum = np.asarray([1.0,1.0])
dimension = 2


#The function to run PES to minimize the target function.
#Parameters: @target_function: the obejective function we want to minimize
#            @x_minimum: the lower bounds for each dimension
#            @x_maximum: the upper bounds for each dimension
#            @dimension: the dimensions of the objective function
#            @number_of_hyperparameter_sets: the number of the samples of the hyperparameters of the kernel we want to draw. 
#                                            It is the M defined in the paper.
#            @number_of_burnin: number of burnins
#            @sampling_method: the method used to sample the posterior distribution of the hyperparameters. User can choose 
#                              'mcmc' or 'hmc'.
#            @number_of_initial_points: the number of samples we want to use as initial observations
#            @number_of_experiments: number of experiments we want to run. For each experiment, we use different randomizations 
#                                    for starting points.
#            @number_of_iterations: number of iterations we want to run for each experiment
#            @number_of_features: the number of features that we would like to use for feature mapping. It is the "m" in the paper.
#            @optimization_method: optimization method used when calling global_optimization function. User can choose any method 
#                                  specified in the scipy.optimize.minimize 
#            @seed: seed specified for randomization
beta_gamma_angles = run_PES(target_function, x_minimum, x_maximum, dimension, number_of_hyperparameter_sets = 100, number_of_burnin = 50, \
        sampling_method = 'mcmc', number_of_initial_points = 10, number_of_experiments = 1, number_of_iterations = 200, \
        number_of_features = 1000, optimization_method = 'SLSQP', seed = 10)




# this function computes the state and susequently the expectation value 
# which we are seeking the minima of

E = expectation_value(beta_gamma_angles, 
                      H,
                      p, 
                      nbr_of_qubits, 
                      initial_state,
                      sigmax)

# final state
gamma_opt= beta_gamma_angles[:p]
beta_opt = beta_gamma_angles[p:2*p]


qaoa_state = state_fast(H, 
                        p, 
                        nbr_of_qubits, 
                        initial_state, 
                        sigmax, 
                        gamma_opt, 
                        beta_opt)
# determine what the optimal value is 
opt_val_idx = 3 # this must be known for the classical optimization problem
prob_of_obtaining_correct_answer_1_shot = np.abs(qaoa_state[3])**2


from plot_energy_landscape import plot_energy_lanscapes

plot_energy_lanscapes(  H,
                            p, 
                            nbr_of_qubits, 
                            initial_state,
                            sigmax, 
                            show_plot=True)
