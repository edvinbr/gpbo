from util import  state_fast, expectation_value 
import matplotlib.pyplot as plt
from pylab import savefig
import numpy as np
from matplotlib import rc as rc
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

def plot_energy_lanscapes(  H,
                            p, 
                            nbr_of_qubits, 
                            initial_state,
                            sigmax, 
                            show_plot=True):
    #=====================================================================================================
    #------------------------- plot the expectation value function for p=1 -------------------------------
    #=====================================================================================================
    fontsize = 20
    figsize=(7,6)
    rc('font',**{'size'   : 16})
    rc('text', usetex=True)

    #plot the energy landscape
    print('Begin to plot')
    f = []
    gamma_nbr_of_points = 80
    beta_nbr_of_points = 40
    gammalin = np.linspace(0, 2*np.pi , gamma_nbr_of_points)
    betalin = np.linspace(0, np.pi , beta_nbr_of_points)
    Z = np.zeros((gamma_nbr_of_points, beta_nbr_of_points))
    gx = 0
    for gamma in gammalin:
        bx = 0
        for beta in betalin:
            expectation_value_function = expectation_value
            Z[gx, bx] = expectation_value_function([gamma, beta],
                                                        H, 
                                                        1, 
                                                        nbr_of_qubits, 
                                                        initial_state, 
                                                        sigmax) 
            bx += 1
        gx += 1

    Beta, Gamma = np.meshgrid(betalin, gammalin)

    # 3D PLOT
    fig = plt.figure(figsize= figsize)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(Beta, 
                            Gamma, 
                            Z,  rstride=1, cstride=1,
                            cmap='viridis', edgecolor='none')

    plt.ylabel(r'$\gamma$', fontsize=fontsize)
    plt.xlabel(r'$\beta$', fontsize=fontsize)

    filename = 'test_3d_1'
    savefig(filename + '.eps')
    savefig(filename + '.png')

    # CONTOUR PLOT
    fig = plt.figure(figsize= figsize)
    plt.contourf(Beta, 
                 Gamma, 
                 Z,  
                 100, 
                 cmap='viridis')
    plt.colorbar()

    plt.ylabel(r'$\gamma$', fontsize=fontsize)
    plt.xlabel(r'$\beta$', fontsize=fontsize)
    ax.ticklabel_format(style='sci',scilimits=(-3,4),axis='both')
    ax.xaxis.major.formatter._useMathText = True

    filename = 'test_contour_1'
    savefig(filename+ '.eps')
    savefig(filename + '.png')


    if show_plot == True:
        plt.show()
