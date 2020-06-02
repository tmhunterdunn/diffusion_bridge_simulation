import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def b_hyp(x, alpha):
    theta = alpha[0]
    return -theta*x/((1 + x**2)**0.5)

def sigma_hyp(x, beta):
    return(beta[0])


def b_OU(x, alpha):
    return(-alpha[0]*x)

def sigma_OU(x, beta):
    return(beta[0])


def simulate_paths_euler(Delta, M, num_paths, x_0, b, sigma):
    """ simulate paths from a diffusion process.

    :b: drift term.
    :sigma: diffusion term.
    :t: vector of time indeces for the path. 
    :x_0: starting point, scalar or vector of length num_paths.
    :num_paths: 
    :returns: TODO

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
    """ simulate paths from a diffusion process.

    :b: drift term.
    :sigma: diffusion term.
    :t: vector of time indeces for the path. 
    :x_0: starting point, scalar or vector of length num_paths.
    :num_paths: 
    :returns: TODO

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
    N = len(t)

    soln = np.zeros((num_paths, N))
    soln[:,0] = 0


    for i in range(1, N):
        delta_t = t[i] - t[i-1]
        soln[:,i] = soln[:,i-1] + np.random.normal(size=(num_paths), scale=delta_t**0.5)

    return soln



def sim_brownian_bridge(t, a, b, num_paths):
    paths = sim_brownian_motion(t, num_paths)

    T = max(t)

    time_mat = np.tile(t/T, [num_paths, 1])
    term_mat = np.tile(paths[:, -1], [ len(t), 1]).T

    bridge = paths - time_mat * term_mat

    bridge = (1- time_mat)*a + time_mat*b + bridge


    return bridge

def simulate_GBM_bridge(u, sigma, T, N,num_paths):
    return sigma*simulate_brownian_bridge(T, N, num_paths) + u*np.arange(N+1)/N

def call_payoff(path, K):
    return max(0, path[-1] - K)

def conv_lookback_call_payoff(path):
    """This is a floating strike lookback call"""
    S_max = np.max(path)
    S_T = path[-1]
    return (S_max - S_T)**3

def lookback_call_payoff(path):
    """This is a floating strike lookback call"""
    S_max = np.max(path)
    S_T = path[-1]
    return S_max - S_T

def asian_call_payoff(path):
    """This is a floating strike arithmetic asian option"""
    A_T = np.mean(path)
    S_T = path[-1]
    return max(A_T - S_T, 0)

def down_and_out_payoff(path, base_payoff, level):
    if np.min(path) <= level:
        return 0
    else:
        return base_payoff(path)

