import numpy as np
import math as m
import cplex

def expectation_value_float(s, cost):
    return (np.dot(s.transpose().conj(), np.multiply(cost, s)).real)[0][0]


def expectation_value(x, *args):
    """
    Compute the expectation value 
    Parameters
    ----------
    x   : np array containing variational parameters(angles) 
        [beta_angles, gamma_angles]
    args: cost (ising spin glass hamiltonian)
          p (iteration level of qaoa)
          q (nbr of qubits)
          s (initial state, i.e. uniform superposition)
          sigmax pauli operator 
    """
    cost = args[0]
    p = args[1]
    q = args[2]
    s = args[3]
    sigmax = args[4]

    gamma = x[0:p]
    beta = x[p:2*p]

    #print (cost, 'p', p, 'q', q, s, sigmax, gamma, beta, x)
    s = state_fast(cost, p, q, s, sigmax, gamma, beta)
    return expectation_value_float(s, cost)


def state_fast(cost, p, q, s, sigmax, gamma, beta):
    """

    Parameters
    ----------
    -------
    """
    for i in range(0, p):
        # Hadamard product, in other words, we do an entrywise product, since
        # the Hamiltonian is diagonal.
        s = np.multiply(np.exp(-1j * gamma[i] * cost), s)

        for j in range(1, q + 1):
            # Construct the rotation matrix and apply it to the state vector.
            # Use fast Kronecker matrix multiplication for matrices
            s = np.cos(beta[i]) * s - 1j * np.sin(beta[i]) * kron(j, q, s)
    return s


def kron(j, q, s):
    p = 2**(j - 1)  # dimension of left identity matrix
    r = 2**(q - j)  # dimension of right identity matrix
    sigmax = np.array([[0, 1], [1, 0]])

    # rearrange s to a (2 x pr) matrix
    s = np.reshape(s, (r, 2, p), order='F')
    s = np.transpose(s, (1, 0, 2))
    s = np.reshape(s, (2, 2**(q - 1)), order='F')

    # actual multiplication
    s = np.dot(sigmax, s)

    # rearrange back the result to a vector
    s = np.transpose(np.reshape(s, (2, p, r)), (1, 0, 2))
    s = np.array([s.flatten()]).transpose()

    return s

def create_ising_hamiltonian(cost_vector, 
                                constraint_matrix, 
                                nbr_of_routes, 
                                nbr_of_flights):

    """
    Parameters
    ----------
    cost_vector of optimization problem, 
    constraint_matrix of optimisation problem
                min sum_{r in routes} c_rx_r, 
                s.t sum_{r in routes} a_{fr} x_r = 1 for all flights
                    x_i in {0, 1}^nbr_of_routes   
    nbr_of_routes in optimization problem 
    nbr_of_flights in optimization problem

    Returns
    -------
    linear_cost_hamiltonian
    quadratic_cost_hamiltonian
    cost_hamiltonian

    of the ising spin glass hamiltonian
    """
    # Create all possible string configurations
    s = np.ones((2**nbr_of_routes, nbr_of_routes)) 
    for i in range(0, 2**nbr_of_routes):
        tmp = np.binary_repr(i, width=nbr_of_routes)
        s[i,:] = list(map(int, tmp))

    # Compute hamiltonian of the linear objective function ----  (UNCONSTRAINED)       
    linear_cost_hamiltonian = np.dot( s, cost_vector)
    # Compute hamiltonian of the constraints              ------  (EXACT COVER)  
    quadratic_cost_hamiltonian =  \
        np.sum(np.power(np.dot(constraint_matrix, s.transpose())-np.ones((nbr_of_flights, 2**nbr_of_routes) ),2), axis=0).reshape(2**nbr_of_routes,1) 
                                
    cost_hamiltonian = linear_cost_hamiltonian + quadratic_cost_hamiltonian 
    return linear_cost_hamiltonian, quadratic_cost_hamiltonian, cost_hamiltonian


def read_data_from_mps(file_location, file_name):
    """
    Reads data from an mps file. The optimization problem is 
    Set-Partitioning:
        min c^Tx, 
        s.t Ax = 1
        x binary integer string in alphabet {0, 1}
    """
    cplex_problem = cplex.Cplex()
    out = cplex_problem.set_results_stream(None)
    out = cplex_problem.set_log_stream(None)

    cplex_problem.read(file_location + file_name)
    
    # get constraint columns 
    columns = cplex_problem.variables.get_cols()
    # define problem size 
    nbr_of_routes = len(columns)
    nbr_of_flights = cplex_problem.linear_constraints.get_num()
    # retrive the cost vector c
    cost_vector = np.array(cplex_problem.objective.get_linear()).reshape(nbr_of_routes,1)
    # construct the constraint matrix A
    constraint_matrix = np.zeros((nbr_of_flights, nbr_of_routes ))
    for r in range(0, nbr_of_routes):
        col = columns[r]
        ind = col.ind
        val = col.val
        for f in range(0, len(val)):
            constraint_matrix[ind[f]][r] = val[f] 

    return cost_vector, constraint_matrix, nbr_of_flights, nbr_of_routes
