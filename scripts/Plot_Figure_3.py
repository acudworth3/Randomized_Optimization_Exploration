import mlrose_hiive as mlrose_hiive
import numpy as np
import itertools
import matplotlib.pyplot as plt

fitness = mlrose_hiive.FourPeaks(t_pct=(2/7))
test_state = np.array([1, 1, 1, 0, 1, 0])
GM_1 = np.array([1, 1, 1,0,0,0,0])
GM_2 = np.array([0,0,0,1,1,1,1])
LM_1 = np.ones(7)
LM_2 = np.zeros(7)

#TODO finish this
n = 7
#https://stackoverflow.com/questions/14931769/how-to-get-all-combination-of-n-binary-value
state_space = np.array([list(i) for i in itertools.product([0, 1], repeat=n)])
fitness_space = np.array([fitness.evaluate(s) for s in state_space])
np.random.shuffle(fitness_space)
plt.scatter(range(len(fitness_space)),fitness_space,s=2.5,color='black',marker='x')

plt.title("Fitness Function\n State Space N = 7 T = 2",fontsize=20)
plt.xlabel("State Space Vectors",fontsize=18)
plt.ylabel("Fitness",fontsize=18)
plt.xticks([],labels=None)
plt.show()


imarker = 1
# problem = mlrose_hiive.DiscreteOpt(length = 12, fitness_fn = fitness, maximize=False, max_val=2)

