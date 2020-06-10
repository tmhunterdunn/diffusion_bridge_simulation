import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pynverse import inversefunc
import scipy.integrate as integrate
from diffusion_utils import sim_paths, sim_brownian_bridge, b_hyp, sigma_hyp
from scipy.optimize import minimize
from scipy.misc import derivative


def b_hyp_prime(x, alpha):

    theta = alpha[0]
    return -theta/((1+x**2)**1.5)

def phi(x, b, b_prime):

    return 0.5*(b(x)**2 + b_prime(x))

def EA1(t, x_0, x_T, b, sample_size, b_prime=None):

    T = t[-1]

    if b_prime is None:
        b_prime = lambda x: derivative(b, x, dx=1e-6)

    M = -minimize(lambda x:-phi(x, b, b_prime), 1).fun


    Z_sample = []
    paths = []
    skels = []


    while len(Z_sample) < sample_size:
        number_of_events = np.random.poisson(T*M)

        if number_of_events == 0:
            Z_sample.append(sim_brownian_bridge(t, x_0, x_T, 1)[0])
            continue

        while True:
            events_x = np.random.uniform(low=0, high=T, size=number_of_events)
            events_x.sort()
            events_y = np.random.uniform(low=0, high=M+1, size=number_of_events)

            skel_time = events_x.copy()
            if skel_time[0] > 0:
                skel_time = np.insert(skel_time, 0, 0)
            if skel_time[-1] < T:
                skel_time = np.append(skel_time, T)

            path = sim_brownian_bridge(skel_time, x_0, x_T, 1)[0]

            phi_vec = np.vectorize(lambda x: phi(x, b, b_prime))

            path_for_phi = path.copy()
            if events_x[0] > 0:
                path_for_phi = path_for_phi[1:]
            if events_x[-1] < T:
                path_for_phi = path_for_phi[:-1] 

            phi_path = phi_vec(path_for_phi)

            N = (events_y > phi_path).sum()

            

            if N == 0:
                paths.append(path)
                skels.append(skel_time)
                break
            
        filled_paths = []
        for i in range(len(path)-1):
            time_range_og = t[np.logical_and(skel_time[i]<=t , skel_time[i+1]>=t)]
            if len(time_range_og) == 0:
                continue


            time_range = time_range_og.copy()

            if time_range[0] > skel_time[i]:
                time_range = np.insert(time_range,0, skel_time[i])
            if time_range[-1] < skel_time[i+1]:
                time_range = np.append(time_range, skel_time[i+1])

            path_chunk = sim_brownian_bridge(time_range, path[i], path[i+1], 1 )[0]
            if time_range_og[0] > skel_time[i]:
                path_chunk = path_chunk[1:]
            if time_range_og[-1] < skel_time[i+1]:
                path_chunk = path_chunk[:-1]

            filled_paths.append(path_chunk)


        Z_sample.append( np.concatenate(filled_paths))

    sample = np.vstack(Z_sample)
    return sample


def h_beta(x, beta, sigma, x_star):

    return integrate.quad( lambda z: 1/sigma(z, beta), x_star, x)[0]

def mu(alpha, beta, b, sigma, y, x_star, sigma_prime=None):

    def h_for_inv(x):
        return(h_beta(x, beta, sigma, x_star))


    h_inv = inversefunc( h_for_inv)(y)

    if sigma_prime is None:
        def sigma_for_grad(x):
            return(sigma(x, beta))
    
        sigma_prime = grad( sigma_for_grad)
        return (b(h_inv, alpha)/sigma(h_inv, beta)) - 0.5*sigma_prime(h_inv)
    else:
        return (b(h_inv, alpha)/sigma(h_inv, beta)) - 0.5*sigma_prime(h_inv, beta)
        


if __name__ == "__main__":

    Delta = 5
    N = 50

    delta = Delta/N

    t = np.arange(N+1)*delta

    x_0 = 2
    x_t = 4

    theta = 1
    sigma_const = 1

    alpha = [theta]
    beta = [sigma_const]
    b = lambda x: b_hyp(x, alpha)
    b_prime = lambda x: b_hyp_prime(x, alpha)
    sigma = lambda x: sigma_hyp(x, beta)


    paths = EA1(t, x_0, x_t, b, 20, b_prime)
    ax = pd.DataFrame(paths.T).plot()
    ax.get_legend().remove()
    ax.set_title("Hyperbolic Bridge Sample")


    plt.show()
