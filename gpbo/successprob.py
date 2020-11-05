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
	#print(H)

	# mixer operator
	sigmax = np.array([[0, 1], [1, 0]])

	initial_state = 1.0/np.sqrt(nbr_states)*np.ones((nbr_states, 1))

	# compute the state for a variational angle for a certain level of p
	# the dimension of the expectation value function is 2*p
	p = int(len(beta_gamma_angles)/2)

	# final state
	#beta_gamma_angles = [(b+1)/2*2*np.pi for b in beta_gamma_angles]
	gamma_opt = [(b+1)/2*2*np.pi for b in beta_gamma_angles[:p]]#(b+1)/2
	beta_opt = [(b+1)/2*np.pi for b in beta_gamma_angles[p:2*p]]
	#gamma_opt= beta_gamma_angles[:p]
	#beta_opt = beta_gamma_angles[p:2*p]

	qaoa_state = state_fast(H, p, nbr_of_qubits, initial_state, sigmax, gamma_opt, beta_opt)
	# determine what the optimal value is
	opt_val_idx = np.argmin(H)#184#3 # this must be known for the classical optimization problem
	prob_of_obtaining_correct_answer_1_shot = np.abs(qaoa_state[opt_val_idx])**2
	E = expectation_value((gamma_opt+beta_opt), H, p, nbr_of_qubits, initial_state,sigmax)
	print(E)
	return prob_of_obtaining_correct_answer_1_shot

#angles = [0.98785665, 0.63002621, 0.62536196, 0.49947186]
#beta_gamma_angles = [(b+1)/2 for b in angles]
#print(qaoaSuccessProb(angles))

print(qaoaSuccessProb([0.9767939819009309,-1,-0.7470681088246521,-1]))
print(qaoaSuccessProb([0.9767939819009309,-0.7470681088246521]))


#[-0.6932115556676391, -0.641477285664184, -0.0027615723835575294, 0.8284715862963159])) #[0.9767939819009309,-0.7470681088246521]))#0.98851643, 0.09585667]))#[-0.9772818585426836, 0.7536042297443648]#[0.7536042297443648, -0.9772818585426836]

#0.9767939819009309, -0.7470681088246521
