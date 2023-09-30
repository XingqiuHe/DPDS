import numpy as np
import random
import math

''' Parameter List
    N = 10   ; number of WDs
    T = 100  ; number of time slots
    d_t = 0.01   ; duration of each slot

    ; d ~ [d_min, d_max], uniform distribution
    ; d = {10, 20, 30, 40, 50}Kb
    d_min = 1Mb   ; lower bound of data size
    d_max = 5Mb   ; upper bound of data size
    d_int = 1Mb   ; interval between neighboring data size
    d_sub = 5Kb   ; data size of each subtask

    ; for the task interval Delta, we consider three different distributions
    pattern = {"geo", "map"}
    ; 1. geometric distribution (i.e. g_i(t) bernoulli)
    p_g = 0.2   ; P(Delta = k) = (1-p_g)^{k-1} * p_g
    ; 2. MAP
    TODO

    kappa = 1000   ; one bit data require kappa CPU cycles
    gamma = 10^{-27}   ; energy-efficiency factor
    eta = gamma * kappa^3 / d_t^2
    h[i][t] = {3,6,9}*10^{-10}   ; channel gain
    sigma^2 = 10^{-9}   ; noise power

    W_max = 10MHz   ; each WD share 1MHz (in expectation)
    E_max[i] = 0.5W
    f_max[i] = 1GHz
    P_max[i] = 1W
'''

''' Variable List
    f[i][t]   ; CPU frequency of WD i at slot t
    d_l[i][t]   ; locally-processed data
    E_l[i][t]   ; energy consumption due to local processing

    r[i][t]   ; wireless transmission rate
    W[i][t]   ; wireless bandwidth
    P[i][t]   ; transmission power
    d_o[i][t]   ; offloaded data
    E_o[i][t]   ; energy consumption due to offloading
'''

class Parameter(object):
    def __init__(self, N=3, T=100, pattern = "geo"):
        super().__init__()

        self.N = N
        self.T = T
        self.d_t = 0.01

        self.pattern = pattern
        self.d_lb = 2e4
        self.d_ub = 5e4
        self.p_g = 0.3

        self.kappa = 1000.0
        self.gamma = 1e-28
        self.eta = self.gamma * self.kappa**3 / self.d_t**2
        self.sigma2 = 1e-11

        # channel model
        self.length = 100
        self.epsilon = 3.8
        self.BS_loc = (self.length/2, self.length/2)
        self.WD_loc_list = [(random.randint(0,self.length), random.randint(0,self.length)) for _ in range(self.N)]
        self.distance = [math.sqrt((self.BS_loc[0]-WD_loc[0])**2 + (self.BS_loc[1]-WD_loc[1])**2) for WD_loc in self.WD_loc_list]
        print(self.distance)
        self.distance = np.array(self.distance)

        self.W_max = 20e6
        self.E_max = [0.1*self.d_t for _ in range(N)]
        self.f_max = [2e9 for _ in range(N)]
        self.P_max = [1.0 for _ in range(N)]
        self.lam_max = [100000 for _ in range(N)]

        self.beta = lambda t: 1 / np.sqrt(t)
        #self.beta = lambda t: 0.0001
        self.theta = lambda t: min(100, 1000/np.log(t+10))
        #self.theta = lambda t: max(10, 10/np.sqrt(t))
        #self.theta = lambda t: 0
        self.lam_init = np.ones(N) * 5000
        #self.lam_init = np.zeros(N)
