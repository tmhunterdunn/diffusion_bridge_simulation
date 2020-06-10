import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def b_hyp(x, alpha):
    """The drift term for a hyperbolic diffusion.

    :x: the state of the process.
    :alpha: vector of parameters.

    """
    theta = alpha[0]
    return -theta*x/((1 + x**2)**0.5)

def sigma_hyp(x, beta):
    """The diffusion term for a hyperbolic diffusion.

    :x: the state of the process.
    :beta: vector of parameters.

    """
    return(beta[0])


def b_OU(x, alpha):
    """The drift term for an Ornstein-Uhlenbeck diffusion.

    :x: the state of the process.
    :alpha: vector of parameters.

    """
    return(-alpha[0]*x)

def sigma_OU(x, beta):
    """The diffusion term for an Ornstein-Uhlenbeck diffusion.

    :x: the state of the process.
    :alpha: vector of parameters.

    """

    return(beta[0])


def simulate_paths_euler(Delta, M, num_paths, x_0, b, sigma):
    """ Simulate paths from a diffusion process on a fixed time grid using the euler method.

    :Delta: Length of time for the simulation
    :M: number of time intervals to be simulated.
    :num_paths: number of paths to be simulated. Integer.
    :x_0: starting point, scalar or vector of length num_paths.
    :b: drift term of the diffusion. Function that takes in and returns a float.
    :sigma: diffusion term of the diffusion. Function that takes in and returns a float.
    :returns: num_paths by M+1 numpy array of paths.

    """
    
    delta_t = Delta/M

    soln = np.zeros((num_paths, M+1))
    soln[:,0] = x_0

    b_vec = np.vectorize(b)
    sigma_vec = np.vectorize(sigma)

    for i in range(1, M+1):
        soln[:,i] = soln[:,i-1] \
                   + b_vec(soln[:,i-1])*delta_t \
                   + sigma_vec(soln[:,i-1])*np.random.normal(size=(num_paths), scale=delta_t**0.5)

    return soln

def sim_paths(b, sigma, t, x_0, num_paths):
    """ simulate paths from a diffusion process on a variable time grid using the euler method.

    :b: drift term. Function that takes in and returns a float.
    :sigma: diffusion term. Function that takes in and returns a float.
    :t: vector of time indeces for the path. 
    :x_0: starting point, scalar or vector of length num_paths.
    :num_paths: the number of paths to be simulated. Integer.
    :returns: num_paths by M+1 numpy array of paths.

    """
    
    N = len(t)

    soln = np.zeros((num_paths, N))
    soln[:,0] = x_0

    b_vec = np.vectorize(b)
    sigma_vec = np.vectorize(sigma)

    for i in range(1, N):
        delta_t = t[i] - t[i-1]
        soln[:,i] = soln[:,i-1] \
                   + b_vec(soln[:,i-1])*delta_t \
                   + sigma_vec(soln[:,i-1])*np.random.normal(size=(num_paths), scale=delta_t**0.5)

    return soln


def simulate_brownian_bridge(T, N, num_paths):
    """ simulate paths from a brownian bridge on a fixed time grid. The bridge starts and ends at zero.

    :T: Time duration of simulation
    :N: Number of time periods for the simulation.
    :num_paths: the number of paths to be simulated. Integer.
    :returns: num_paths by N+1 numpy array of paths.

    """
    soln = np.zeros((num_paths, N+1))
    soln[:,0] = 0

    for i in range(int(np.log(N)/np.log(2))):
        for shift_no in range(2**i):
            beg = (shift_no*N//(2**i))
            mid = (shift_no*N//(2**i)) + N//(2**(i+1))
            end = (1+shift_no)*N//(2**i)
            soln[:,mid] = np.random.normal((soln[:,beg] + soln[:,end])/2, 0.25*T/(2**i), num_paths)

    return soln

def sim_brownian_motion(t, num_paths):
    """ Simulate paths from brownian motion on a variable time grid.

    :t: vector of time indeces for the path.
    :num_paths: the number of paths to be simulated. Integer.

    :returns: num_paths by len(t) numpy array of paths.


    """
    N = len(t)

    soln = np.zeros((num_paths, N))
    soln[:,0] = 0


    for i in range(1, N):
        delta_t = t[i] - t[i-1]
        soln[:,i] = soln[:,i-1] + np.random.normal(size=(num_paths), scale=delta_t**0.5)

    return soln



def sim_brownian_bridge(t, a, b, num_paths):
    """ simulate paths from a brownian bridge on a variable time grid. The bridge starts at a and ends at b.

    :t: vector of time indeces for simulation
    :a: the starting point of the bridge. Float.
    :b: the end point of the bridge. Float.

    :returns: num_paths by len(t) numpy array of paths.

    """
    paths = sim_brownian_motion(t, num_paths)

    T = max(t)

    time_mat = np.tile(t/T, [num_paths, 1])
    term_mat = np.tile(paths[:, -1], [ len(t), 1]).T

    bridge = paths - time_mat * term_mat

    bridge = (1- time_mat)*a + time_mat*b + bridge


    return bridge





