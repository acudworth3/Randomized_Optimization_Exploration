import mlrose_hiive
import numpy as np
from matplotlib import pyplot as plt
from math import inf
import time
from tqdm import tqdm


#get time and randomize

init_state = np.random.randint(0,2,25)
fitness_P = mlrose_hiive.FourPeaks(t_pct=0.15)
best_state = np.zeros(len(init_state))
best_state[0:round(0.15*len(init_state)+1)] = 1
peak_best_score = fitness_P.evaluate(best_state)
P_problem = mlrose_hiive.DiscreteOpt(length = len(init_state), fitness_fn = fitness_P, maximize=True,max_val=2)

#K color
#TODO make Australia map
edges = [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (2, 5), (3, 4),(4,5)] #Australia
# edges = [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (2, 5), (3, 4), (4, 5), (5, 6), (6, 7), (6, 8), (7, 9), (8, 9), (8, 10), (8, 11), (9, 10), (10, 11), (11, 12), (12, 13), (12, 14), (13, 15), (14, 15), (14, 16), (14, 17), (15, 16), (16, 17), (17, 18), (18, 19), (18, 20), (19, 21), (20, 21), (20, 22), (20, 23), (21, 22), (22, 23), (23, 24), (24, 25), (24, 26), (25, 27), (26, 27), (26, 28), (26, 29), (27, 28), (28, 29), (29, 30)]
fitness_k = mlrose_hiive.MaxKColor(edges)
# init_state = np.array([0, 1, 0, 1, 1])
K_problem = mlrose_hiive.DiscreteOpt(length = max(max(edges))+1, fitness_fn = fitness_k, maximize=False, max_val=3)



#8-Queens
fitness_Q = mlrose_hiive.Queens()
#OPti_object
Q_problem = mlrose_hiive.DiscreteOpt(length = 8, fitness_fn = fitness_Q, maximize=False, max_val=8)
# init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])

#TODO paramaetrize
problem_dict = {'N-Queens':Q_problem,
                '4-Peaks':P_problem,
                'K-Color':K_problem}


four_peaks_max_state =[0,0,0,1,1,1,1]
init_state_dict = {'N-Queens':np.random.randint(0,len(Q_problem.state),len(Q_problem.state)),
                '4-Peaks':P_problem.state,
                'K-Color':K_problem.state}



class plt_obj():
    """ Trying to Stay originized"""
    def __init__(self,n=None):
        self.plot_dict = {}
        self.peak_best_score = 1
        self.RHC = True
        self.SA = True
        self.GA = True
        self.MC = True
        self.axes_dict = {0:'N-Queens',1:'4-Peaks',2:'K-Color'}

    def process_results(self,matrx_obj,problem,algo_name):
        if problem == '4-Peaks':
            # GA_avg_fit_v_iter = GA_avg_fit_v_iter / peak_best_score
            # GA_avg_fit_v_iter *=-1
            # GA_avg_fit_v_iter += 1
            matrx_obj = matrx_obj / self.peak_best_score
            matrx_obj *= -1
            matrx_obj += 1
        else:
            # GA_avg_fit_v_iter = GA_avg_fit_v_iter / max(GA_avg_fit_v_iter)
            matrx_obj = matrx_obj / np.max(matrx_obj)

        avg_fit_v_iter = np.mean(matrx_obj,axis=0)

        med_fit_v_iter = np.median(matrx_obj,axis=0)
        std_fit_v_iter = np.std(matrx_obj,axis=0)

        self.plot_dict[algo_name+'_'+problem+'_std'] = std_fit_v_iter
        self.plot_dict[algo_name+'_'+problem+'_avg'] = avg_fit_v_iter
        self.plot_dict[algo_name+'_'+problem+'_med'] = med_fit_v_iter

    def fig_1(self):
        fig, axes = plt.subplots(1, 3, figsize=(20, 5.05),sharey=True)
        fig.suptitle('Averaged Normalized Fitness vs Function Iteration',y=1.0,fontsize=25)
        fig.subplots_adjust(top=0.5)
        #TODO adjust axes x limits
        axes[1].set_xlabel('Iterations',fontsize=20)
        axes[0].set_ylabel('Normalized Average Fitness',fontsize=20)
        axes[0].set_xlim(0, 100)
        axes[1].set_xlim(0, 100)
        axes[2].set_xlim(0, 25)
        # axes[0].set_title("N-Queens")
        # axes[1].set_title("N-Queens")
        # Plot N-Queens

        for idx in self.axes_dict.keys():
            axes[idx].set_title(self.axes_dict[idx],fontsize=20)
            axes[idx].grid()
            axes[idx].set_ylim(-0.1, 1.1)
            if self.GA:
                axes[idx].fill_between(range(len(self.plot_dict['GA_'+self.axes_dict[idx]+'_avg'])),
                                       self.plot_dict['GA_'+self.axes_dict[idx]+'_avg'] - self.plot_dict['GA_'+self.axes_dict[idx]+'_std'],
                                       self.plot_dict['GA_'+self.axes_dict[idx]+'_avg'] + self.plot_dict['GA_'+self.axes_dict[idx]+'_std'], alpha=0.1,
                                       color="b")
                axes[idx].plot(range(len(self.plot_dict['GA_'+self.axes_dict[idx]+'_avg'])), self.plot_dict['GA_'+self.axes_dict[idx]+'_avg'], 'o--', markersize=2.5,linewidth=0.5,
                               color="b",
                               label="Gen_Alg")

            if self.RHC:
                axes[idx].fill_between(range(len(self.plot_dict['RHC_'+self.axes_dict[idx]+'_avg'])),
                                       self.plot_dict['RHC_'+self.axes_dict[idx]+'_avg'] - self.plot_dict['RHC_'+self.axes_dict[idx]+'_std'],
                                       self.plot_dict['RHC_'+self.axes_dict[idx]+'_avg'] + self.plot_dict['RHC_'+self.axes_dict[idx]+'_std'], alpha=0.1,
                                       color="g")
                axes[idx].plot(range(len(self.plot_dict['RHC_'+self.axes_dict[idx]+'_avg'])), self.plot_dict['RHC_'+self.axes_dict[idx]+'_avg'], 'o--', markersize=2.5,linewidth=0.5,
                               color="g",
                               label="RHC")

            if self.SA:
                axes[idx].fill_between(range(len(self.plot_dict['SA_'+self.axes_dict[idx]+'_avg'])),
                                       self.plot_dict['SA_'+self.axes_dict[idx]+'_avg'] - self.plot_dict['SA_'+self.axes_dict[idx]+'_std'],
                                       self.plot_dict['SA_'+self.axes_dict[idx]+'_avg'] + self.plot_dict['SA_'+self.axes_dict[idx]+'_std'], alpha=0.1,
                                       color="c")
                axes[idx].plot(range(len(self.plot_dict['SA_'+self.axes_dict[idx]+'_avg'])), self.plot_dict['SA_'+self.axes_dict[idx]+'_avg'], 'o--', markersize=2.5,linewidth=0.5,
                               color="c",
                               label="Sim_Aneal")

            if self.MC:
                axes[idx].fill_between(range(len(self.plot_dict['MC_'+self.axes_dict[idx]+'_avg'])),
                                       self.plot_dict['MC_'+self.axes_dict[idx]+'_avg'] - self.plot_dict['MC_'+self.axes_dict[idx]+'_std'],
                                       self.plot_dict['MC_'+self.axes_dict[idx]+'_avg'] + self.plot_dict['MC_'+self.axes_dict[idx]+'_std'], alpha=0.1,
                                       color="y")
                axes[idx].plot(range(len(self.plot_dict['MC_'+self.axes_dict[idx]+'_avg'])), self.plot_dict['MC_'+self.axes_dict[idx]+'_avg'], 'o--', markersize=2.5,linewidth=0.5,
                               color="y",
                               label="Mimic")

        plt.legend(loc="best",fontsize=15)


        fig.tight_layout()
        plt.show()
        
        plt.savefig('fig1_iterv_fitn_normed'+str(np.random.randint(0,2*25,1))+'.png')


fig1_obj = plt_obj()
fig1_obj.peak_best_score = peak_best_score
fig1_obj.MC = True
fig1_obj.RHC = True
fig1_obj.SA = True
fig1_obj.GA = True

for problem in problem_dict.keys():
    iter_plot_size = 10
    GA_test_obj = []
    time_vec = []
    if fig1_obj.MC:
        print("\nMC: "+problem)

        MC_test_obj = np.array([mlrose_hiive.mimic(problem_dict[problem],pop_size = 20, keep_pct=0.50, max_attempts = 2**30,
                                                              max_iters = 100,
                                                              random_state = np.random.randint(0,2**30),curve=True) for i in tqdm(range(iter_plot_size))]) #TODO account for randomness


        # if problem != 'K-Color':
        #     MC_test_obj = np.array(
        #         [mlrose_hiive.mimic(problem_dict[problem], pop_size=2000, keep_pct=0.50, max_attempts=2 ** 30,
        #                             max_iters=100,
        #                             random_state=np.random.randint(0, 2 ** 30), curve=True) for i in
        #          tqdm(range(iter_plot_size))])  # TODO account for randomness
        MC_crv_matrix = np.array([MC_test_obj[:,2][idx] for idx in range(len(MC_test_obj[:,2]))])
        # else:
        #     MC_crv_matrix = GA_crv_matrix*0+1
        fig1_obj.process_results(MC_crv_matrix, problem, algo_name='MC')


    if fig1_obj.RHC:
        print("\nRHC: "+problem)
        RHC_test_obj = np.array([mlrose_hiive.random_hill_climb(problem_dict[problem], max_attempts = 2**32,
                                                              max_iters = 100, init_state = init_state_dict[problem],restarts = 20,
                                                              random_state = np.random.randint(0,2**30),curve=True) for i in tqdm(range(iter_plot_size))]) #TODO account for randomness

        RHC_crv_matrix = np.array([RHC_test_obj[:, 2][idx] for idx in range(len(RHC_test_obj[:, 2]))])
        fig1_obj.process_results(RHC_crv_matrix, problem, algo_name='RHC')

    if fig1_obj.SA:
        print("\nSA: "+problem)
        # Define decay schedule (sim annealing only)
        # schedule = mlrose_hiive.ExpDecay(init_temp=1.0, exp_const=0.0005, min_temp=0.00001)
        # schedule = mlrose_hiive.ArithDecay()
        schedule = mlrose_hiive.GeomDecay(init_temp=1.0, decay=0.5, min_temp=0.001)

        SA_test_obj = np.array([mlrose_hiive.simulated_annealing(problem_dict[problem], schedule = schedule, max_attempts = 2**32,
                                                              max_iters = 100, init_state = init_state_dict[problem],
                                                              random_state = np.random.randint(0,2**30),curve=True) for i in tqdm(range(iter_plot_size))]) #TODO account for randomness

        SA_crv_matrix = np.array([SA_test_obj[:,2][idx] for idx in range(len(SA_test_obj[:,2]))])
        fig1_obj.process_results(SA_crv_matrix, problem, algo_name='SA')

    if fig1_obj.GA:
        print("\nGA: "+problem)
        GA_test_obj = np.array([mlrose_hiive.genetic_alg(problem_dict[problem],pop_size = 20, mutation_prob=0.01, max_attempts = 2**30,
                                                          max_iters = 100,random_state = np.random.randint(0,2**30),curve=True) for i in tqdm(range(iter_plot_size))]) #TODO account for randomness

        GA_crv_matrix = np.array([GA_test_obj[:, 2][idx] for idx in range(len(GA_test_obj[:, 2]))])
        fig1_obj.process_results(GA_crv_matrix, problem, algo_name='GA')

        #TODO remove K-Color

fig1_obj.fig_1()



#TODO
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
# https://matplotlib.org/gallery/api/two_scales.html
#TRY this plot
#time, fitness calls, and best fitness all on one axis


