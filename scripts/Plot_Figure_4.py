import mlrose_hiive
import numpy as np
import time
from matplotlib import pyplot as plt

#get time and randomize


# 4 PEAKS
N_P=7
P_rand_state = np.random.randint(0,2,N_P)
fitness_P = mlrose_hiive.FourPeaks(t_pct=0.15)
fitness_P.evaluate(P_rand_state)
# P_problem = mlrose_hiive.DiscreteOpt(length = len(init_state), fitness_fn = fitness_P, maximize=True,max_val=2)

#K color
#TODO make Australia map
N_K = 1
base_edges = [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (2, 5), (3, 4),(4,5)] #Australia
edges = []
for au_idx in range(N_K):
    for edge in base_edges:
        edge = (edge[0]+5*au_idx,edge[1]+5*au_idx)
        edges.append(edge)

fitness_k = mlrose_hiive.MaxKColor(edges)
# init_state = np.array([0, 1, 0, 1, 1])
K_problem = mlrose_hiive.DiscreteOpt(length = max(max(edges))+1, fitness_fn = fitness_k, maximize=False, max_val=3)
fitness_k.evaluate(K_problem.state)

# Define decay schedule (sim annealing only)
schedule = mlrose_hiive.ExpDecay()


#8-Queens
fitness_Q = mlrose_hiive.Queens()
N_Q = 8
#OPti_object
# Q_problem = mlrose_hiive.DiscreteOpt(length = 8, fitness_fn = fitness_Q, maximize=False, max_val=8)

# timeit.timeit(fitness_Q.evaluate(Q_rand_state), number=100000)

class var_input_size():
    def __init__(self,N_Q = 8,N_P=7,N_K=1):
        self.N_Q = N_Q
        self.N_K = N_K
        self.N_P = N_P


    def init_optis(self,K=False):
        self.Q_rand_state = np.random.randint(0,self.N_Q+1,self.N_Q)
        self.fitness_Q = mlrose_hiive.Queens()
        self.P_rand_state = np.random.randint(0,2,self.N_P)
        self.fitness_P = mlrose_hiive.FourPeaks(t_pct=0.15)
        self.eval_cnt = 1
        if K:
            base_edges = [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (2, 5), (3, 4),(4,5)] #Australia
            edges = []
            for au_idx in range(N_K):
                for edge in base_edges:
                    edge = (edge[0]+5*au_idx,edge[1]+5*au_idx)
                    edges.append(edge)
            self.K_rand_state = np.random.randint(0,4,max(max(edges))+1)
            self.fitness_K = mlrose_hiive.MaxKColor(edges)

    def time_Q(self,max_N):
        self.Q_time = np.ones((2,max_N))
        for N_Q in range(2,max_N+1):
            self.N_Q = N_Q
            self.init_optis()
            start = time.time()
            [self.fitness_Q.evaluate(self.Q_rand_state) for i in range(0,self.eval_cnt)]
            delta_t = time.time()-start
            self.Q_time[0,N_Q-1] *= N_Q
            self.Q_time[1,N_Q-1] *= delta_t

        self.Q_time = self.Q_time[:, 2:]

    def time_P(self,max_N):
        self.P_time = np.ones((2,max_N))
        for N_P in range(2,max_N+1):
            self.N_P = N_P
            self.init_optis()
            start = time.time()
            [self.fitness_P.evaluate(self.P_rand_state) for i in range(0,self.eval_cnt)]
            delta_t = time.time()-start
            self.P_time[0,N_P-1] *= N_P
            self.P_time[1,N_P-1] *= delta_t

        self.P_time = self.P_time[:, 2:]

    def time_K(self,max_N):
        self.K_time = np.ones((2,max_N))
        for N_K in range(2,max_N+1):
            self.N_K = N_K
            self.init_optis(K=True)
            start = time.time()
            [self.fitness_K.evaluate(self.K_rand_state) for i in range(0,self.eval_cnt)]
            delta_t = time.time()-start
            self.K_time[0,N_K-1] *= N_K
            self.K_time[1,N_K-1] *= delta_t
        self.K_time = self.K_time[:,2:]

time_plt_obj = var_input_size()
time_plt_obj.time_Q(1000)
time_plt_obj.time_P(5000)
time_plt_obj.time_K(5000)

# plt.scatter(time_plt_obj.Q_time[0,:],time_plt_obj.Q_time[1,:])
plt.scatter(time_plt_obj.Q_time[0,:],time_plt_obj.Q_time[1,:],label='N_Queens',s=2.5,marker='o')
plt.scatter(time_plt_obj.P_time[0,:],time_plt_obj.P_time[1,:],label='4_Peaks',s=2.5,marker='s')
plt.scatter(time_plt_obj.K_time[0,:],time_plt_obj.K_time[1,:],label='K_Colors',s=2.5,marker='^')
plt.legend()
plt.show()
marker = 1
# time_plt_obj.init_optis()