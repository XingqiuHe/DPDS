# +
import numpy as np
import sys
import random
import Parameter as Parameter

import pdb


# -

class Environment(object):
    def __init__(self, param: Parameter):
        super().__init__()
        self.param = param
        # self.reset()

    def reset(self):
        param = self.param

        self.timer = 0
        self.nTask = 0 #number of generated tasks
        self.next_gen = [0 for _ in range(param.N)]
        self.last_datasize = 0
        self.last_delta = np.zeros(param.N, dtype=int)

        # states
        self.d_r = np.zeros(param.N)
        self.a = np.zeros(param.N, dtype=int)
        self.delta = np.zeros(param.N, dtype=int)
        self.q = np.zeros(param.N, dtype=int)
        self.h = self.new_channel_gain()
        self.g = np.zeros(param.N, dtype=int)
        self.q_set = [[] for _ in range(param.N)]
        self.acc_E = np.zeros(param.N) # accumulated energy consumption

        # statistics
        self.E_stat = [0]
        self.A_stat = [0]

        return np.transpose(np.vstack((self.d_r, self.a, self.q, self.h)))

    def step(self, action):
        f = action[:, 0] * self.param.f_max
        P = action[:, 1] * self.param.P_max
        W = action[:, 2] * self.param.W_max
        #assert np.isclose(np.sum(action[:,2]), 1.)
        if not np.isclose(np.sum(action[:,2]), 1.):
            pdb.set_trace()

        if np.isnan(f).any() or np.isnan(P).any() or np.isnan(W).any():
            pdb.set_trace()

        d = f * self.param.d_t / self.param.kappa + \
                self.param.d_t * W * np.log2(1+P*self.h/self.param.sigma2)
        E = self.param.gamma * f**3 * self.param.d_t + P * self.param.d_t
        #pdb.set_trace()
        self.update(d, E)

        return np.transpose(np.vstack((self.d_r, self.a, self.q, self.h))), E

    def update(self, d, E):
        self.timer += 1
        # at the beginning of each slot, observe current channel gain and previous task generation
        self.h = self.new_channel_gain()
        self.g = self.new_task_generation(self.delta)
        # update system states according to previous decisions
        d = np.minimum(d, self.d_r)
        self.d_r = self.d_r - d
        self.acc_E = self.acc_E + E
        for i in range(self.param.N):
            assert self.q[i] == len(self.q_set[i])
            if d[i] > 0 and self.d_r[i] == 0: # HOL task is completed
                self.q_set[i].pop(0) # remove the HOL task
                self.q[i] = self.q[i] - 1
                if self.q[i] > 0:
                    self.d_r[i] = self.q_set[i][0].datasize
                    self.a[i] = self.timer - self.q_set[i][0].generationTime
                else:
                    self.a[i] = 0
            elif self.d_r[i] > 0: # HOL task is not completed
                self.a[i] = self.a[i] + 1
            else: # d_r[i] == 0 and d[i] == 0, then the queue must be empty in last slot
                #assert self.q[i] - self.g[i] == 0
                if self.q[i] - self.g[i] != 0:
                    print(self.q[i])
                    print(self.g[i])
                    pdb.set_trace()
                    assert False
                if self.q[i] > 0:
                    self.d_r[i] = self.q_set[i][0].datasize
                    self.a[i] = self.timer - self.q_set[i][0].generationTime
                else:
                    self.a[i] = 0
        #self.d_r = round_and_check(self.d_r)
        self.E_stat.append(np.sum(E)/self.param.N)
        self.A_stat.append(np.sum(self.a)/self.param.N)

    def new_channel_gain(self):
        rand_expo = [random.expovariate(1.0) for _ in range(self.param.N)]
        rand_expo = np.array(rand_expo)
        h = 1e-3 * self.param.distance**(-self.param.epsilon) * rand_expo
        return h

    def new_task_generation(self, old_delta):
        # remember to minus the timer by 1, because we are observing the task generation of the previous slot
        timer = self.timer - 1
        g = np.zeros(self.param.N)
        for i in range(self.param.N):
            if self.next_gen[i] == timer:
                g[i] = 1
                self.last_delta[i] = self.delta[i]
                self.delta[i] = 1 # reset delta
                # generate new task
                self.last_datasize = random.randint(self.param.d_lb, self.param.d_ub)
                new_task = Task(self.nTask, timer, self.last_datasize, i)
                self.nTask += 1
                # remember to update q[i] and q_set[i]
                self.q_set[i].append(new_task)
                self.q[i] += 1
                if self.param.pattern == "geo":
                    interval = np.random.geometric(self.param.p_g)
                elif self.param.pattern == "map":
                    sys.exit("MAP not implemented yet!")
                else:
                    sys.exit("arrival pattern not implemented yet!")
                assert interval > 0
                self.next_gen[i] += interval
            else:
                self.delta[i] += 1
        return g

class Task(object):
    def __init__(self, index, gt, size, wd):
        self.index = index
        self.generationTime = gt
        self.datasize = size
        self.generationDevice = wd
