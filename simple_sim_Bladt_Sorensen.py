import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from diffusion_utils import sim_paths, simulate_paths_euler, b_hyp, sigma_hyp


def sim_crossing(t, x_0, x_T, b, sigma):
    """ Simulate a crossing between independent forward and backward diffusions.

    :t: vector of time indeces for the simulation.
    :x_0: Starting point of the forward diffusion.
    :X_T: Starting point of the backward diffusion.
    :b: The drift term of the diffusion. Function that takes in and returns a float.
    :sigma: The diffusion term of the diffusion. Function that takes in and returns a float.

    :returns: tupple of length 5 containing, the bridge, forward diffusion, backward diffusion, crossing points and independent diffusion that hit the bridge.

    """
    while True:

        paths = sim_paths(b, sigma, t, [x_0, x_T], 2)
        M = len(t)-1

        Y_1 = paths[0,:]
        Y_2 = paths[1,:]
        cross_points = []

        for i in range(M):
            if ((Y_1[i] >= Y_2[M-i]) and (Y_1[i+1] <= Y_2[M-(i+1)])) or ((Y_1[i] <= Y_2[M-i]) and (Y_1[i+1] >= Y_2[M-(i+1)])):
                cross_points.append(i)

        if len(cross_points) > 0:
            Z = np.zeros(M+1)
            tau_minus = cross_points[0]
            nu = tau_minus + 1
            Z[:nu] = Y_1[:nu]
            Z[nu:] = Y_2[::-1][nu:]
            indep_diff = np.zeros(M+1)
            indep_diff[:nu] = Y_2[::-1][:nu]
            indep_diff[nu:] = Y_1[nu:]
            return (Z, Y_1, Y_2, cross_points, indep_diff)

        

def sim_rho_delta_b_diffusions(t, b, num_paths, alpha, sigma):
    """ Simulate a diffusion process with initial distribution corresponding to the terminal distribution of an independent process if it were started at the point b.


    :t: vector of time indeces for the simulation.
    :b: the starting point of the independent process.
    :num_paths: the number of paths to be simulated. Integer.
    :alpha: The drift term of the diffusion. Function that takes in and returns a float.
    :sigma: The diffusion term of the diffusion. Function that takes in and returns a float.

    :returns: num_paths by M + 1 numpy array of paths. 


    """

    M = len(t)
    new_t = np.concatenate([t[:-1],t + t[-1]])

    paths = sim_paths(alpha, sigma, new_t, b, num_paths)
    
    return(paths[:, M:])

def paths_intersect(p1, p2):
    """ Check if two paths intersect.

    :p1: the first path. array like.
    :p2: the second path. array like.

    :returns: tuple with boolean first element and numpy array of crossing points for second element.

    """
    cross_points = []
    for i in range(min(len(p1), len(p2)) - 1):
        if ((p1[i] >= p2[i]) and (p1[i+1] <= p2[i+1])) or ((p1[i] <= p2[i]) and (p1[i+1] >= p2[i+1])):
            cross_points.append(i)

    if len(cross_points) > 0:
        return (True, cross_points)
    else:
        return (False, cross_points)



def estimate_rho_delta(t, x, b, N, alpha, sigma):
    """ Estimate the reciprocal of the probability that the path x is hit by an independent rho_delta_b diffusion.

    :t: vector of time indeces for the diffusion paths.
    :x: the path.
    :b: the b for the rho_delta_b diffusion.
    :N: number of monte-carlo simulations to be performed for the estimate.
    :alpha: The drift term of the diffusion process. Function that takes and returns a float.
    :sigma: The diffusion term of the diffusion process. Function that takes and returns a float.

    :returns: float. The probability that the path x is hit by and independent rho_delta_b diffusion.

    
    """

    sum_T_j = 0
    for j in range(N):
        T_j = 0

        while True:
            T_j += 1
            Y = sim_rho_delta_b_diffusions(t, b, 1, alpha, sigma)[0]
            if paths_intersect(x, Y)[0]:
                break

        sum_T_j += T_j



    return(sum_T_j/N)




def MH_sampler(t, x_0, x_T, b, sigma, sample_size):
    """ Metropolis-Hastings algorithm for sampling diffusion bridges.

    :t: vector of time indeces for the simulation.
    :x_0: the starting point of the bridge.
    :x_T: the ending point of the bridge.
    :b: The drift term of the diffusion process. Function that takes and returns a float.
    :sigma: The diffusion term of the diffusion process. Function that takes and returns a float.
    :sample_size: The number of paths to be sampled. Integer.

    :returns: sample_size by len(t) numpy array of paths.


    """

    Delta = t[-1]

    burn_in_period = 10
    num_thrown_away = 0


    Z_sample = []

    Z_prev = sim_crossing(t, x_0, x_T, b, sigma)[0]
    rho_delta_prev = estimate_rho_delta(t, Z_prev, x_T, 30, b, sigma)

    while len(Z_sample) < sample_size:

        Z_prop = sim_crossing(t, x_0, x_T, b, sigma)[0]
        rho_delta = estimate_rho_delta(t, Z_prop, x_T, 30, b, sigma)
        r_hat = rho_delta/rho_delta_prev
        alpha_hat = min(1, r_hat)
        U = np.random.uniform()
        
        if num_thrown_away >= burn_in_period:
            if U <= alpha_hat:
                Z_sample.append(Z_prop)
                Z_prev = Z_prop
                rho_delta_prev = rho_delta
            else:
                Z_sample.append(Z_prev)
        else:
            num_thrown_away += 1
            


    sample = np.vstack(Z_sample)
    return(sample)

if __name__ == '__main__':
    Delta = 5
    N = 50

    delta = Delta/N

    t = np.arange(N+1)*delta
    M = len(t) - 1

    x_0 = 2
    x_T = 4

    theta = 1
    sigma_const = 1

    alpha = [theta]
    beta = [sigma_const]
    b = lambda x: b_hyp(x, alpha)
    sigma = lambda x: sigma_hyp(x, beta)

    Z, Y_1, Y_2, cross_points, indep_diff = sim_crossing(t, x_0, x_T, b, sigma)
    fig, ax = plt.subplots()
    forward_diff = Y_1
    backward_diff = Y_2[::-1]
    bridge = Z
    tau = t[cross_points[0] + 1]
    ymin = min(forward_diff.min(), backward_diff.min())
    ymin = ymin - 0.2*abs(ymin)
    ymax = max(forward_diff.max(), backward_diff.max())
    ymax = ymax + 0.2*abs(ymax)
    ax.plot(t, forward_diff, label="Forward diffusion", color='g')
    ax.plot(t, backward_diff, label="Backward diffusion", color='b')
    ax.set_ylim(ymin=ymin, ymax=ymax)
    ax.set_xlim(xmin=0, xmax=Delta)
    ax.set_xticks((tau,))
    ax.vlines(tau, linestyles="dashed", ymin=ymin, ymax=ymax)
    ax.set_xticklabels(("$\\tau$",), fontsize=20)
    ax.legend(loc=2, fontsize=10)
    ax_copy = ax.twinx()
    ax_copy.plot(t, Z, label='Bridge', marker='.', markeredgecolor='r', markerfacecolor='None', linestyle = 'None')
    ax_copy.plot(t, indep_diff, label="Independent diffusion", marker='x', markerfacecolor="None", linestyle='None', markeredgecolor='purple')
    ax_copy.set_yticks((x_T,))
    ax_copy.set_yticklabels(("$b$",), fontsize=20)
    ax_copy.legend(loc=1, fontsize=10)

    ax_copy.set_ylim(ymin=ymin, ymax=ymax)
    ax_copy.set_xlim(xmin=0, xmax=Delta)
    ax.set_yticks((x_0,))
    ax.set_yticklabels(("$a$",), fontsize=20)
    plt.tight_layout()
    

    paths = MH_sampler(t, x_0, x_T, b, sigma, 20)
    
    ax = pd.DataFrame(paths.T).plot()
    ax.get_legend().remove()
    ax.set_title("Hyperbolic Bridge Sample")
    plt.show()
