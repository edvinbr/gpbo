from examples.util import *
import numpy as np

def qaoaSuccessProb(beta_gamma_angles):
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
	cost_hamiltonian] =  create_ising_hamiltonian(c, A, nbr_of_qubits, nbr_of_flights)

	# Exact_cover:
	H = quadratic_cost_hamiltonian

	# mixer operator
	sigmax = np.array([[0, 1], [1, 0]])

	initial_state = 1.0/np.sqrt(nbr_states)*np.ones((nbr_states, 1))

	# compute the state for a variational angle for a certain level of p
	# the dimension of the expectation value function is 2*p
	p = int(len(beta_gamma_angles)/2)

	# final state
	#beta_gamma_angles = [(b+1)/2*2*np.pi for b in beta_gamma_angles]
	gamma_opt = [b*2*np.pi for b in beta_gamma_angles[:p]]#(b+1)/2
	beta_opt = [b*np.pi for b in beta_gamma_angles[p:2*p]]
	#gamma_opt= beta_gamma_angles[:p]
	#beta_opt = beta_gamma_angles[p:2*p]

	qaoa_state = state_fast(H, p, nbr_of_qubits, initial_state, sigmax, gamma_opt, beta_opt)
	# determine what the optimal value is
	opt_val_idx = 3 # this must be known for the classical optimization problem
	prob_of_obtaining_correct_answer_1_shot = np.abs(qaoa_state[opt_val_idx])**2
	return prob_of_obtaining_correct_answer_1_shot


print(qaoaSuccessProb([0.98851643, 0.09585667]))#[-0.9772818585426836, 0.7536042297443648]#[0.7536042297443648, -0.9772818585426836]