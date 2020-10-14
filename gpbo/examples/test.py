from util import *
import gpbo
import scipy as sp
import numpy as np
import os
from datetime import datetime
import sys

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
p = 2

#noise
s = 0.5

def f(beta_gamma_angles,**ev):
    gamma = [(b+1)/2*2*np.pi for b in beta_gamma_angles[:p]]
    beta = [(b+1)/2*np.pi for b in beta_gamma_angles[p:2*p]]
    c = 1.

    E = expectation_value((gamma+beta), H, p, nbr_of_qubits, initial_state,sigmax)
    # noise
    n = sp.random.normal() * s
    # we want to check the noiseless value when evaluating performance
    if 'cheattrue' in ev.keys():
      if ev['cheattrue']:
        n = 0
    print('f inputs x:{} outputs y:{} (n:{})'.format(beta_gamma_angles, E + n, n))
    return E + n, c, dict()

    
#beta = np.pi/2
#gamma = np.pi/3
#beta_gamma_angles = np.array([gamma, beta])
#print(beta_gamma_angles)

timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
C = gpbo.core.config.switchdefault(f, p*2, 10, 250, s, 'results/qaoa', str(p*2) + 'Dqaoa-instance_8_0_sigma5_1_noise' + str(s) + '-' + timestamp + '.csv') #500*(p+1)
C.choosepara['regretswitch'] = 1e-2
C.choosepara['pvetol'] = 1e-2
C.aqpara[1]['tol']=None

out = gpbo.search(C, False)
print(out)
print('True value for noisy functions')
f(out[0],cheattrue=True)[0]

beta_gamma_angles = out[0]

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

#plot_energy_lanscapes(  H,
#                            p, 
#                            nbr_of_qubits, 
#                            initial_state,
#                            sigmax, 
#                            show_plot=True)
