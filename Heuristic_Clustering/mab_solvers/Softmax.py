import numpy as np
from numpy.random import choice

from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType

from Metric import metric
from mab_solvers.MabSolver import MabSolver


class Softmax(MabSolver):
    def __init__(self, action, params, is_fair=True):
        MabSolver.__init__(self, action, params)
        self.num_algos = params.num_algos
        self.rewards = np.zeros(params.num_algos)
        self.n = np.array([1] * self.num_algos)
        self.tau = params.tau
        self.is_fair = is_fair
        # self.name = "softmax" + str(tau * 10)

    # def initialize(self, f):
    #     print("\nInit Softmax with tau = " + str(self.tau))
    #     # start = time.time()
    #     for i in range(0, Constants.num_algos):
    #         # self.r[i] = self.action.apply(i, f, i)
    #         # instead af calling smac ^ get some randon config and calculate reward for i-th algo:
    #
    #         ex = self.action  # AlgoExecutor
    #         t = ClusteringArmThread(ex.clu_algos[i], ex.metric, ex.X, ex.seed)
    #         random_cfg = t.clu_cs.sample_configuration()
    #
    #         # run on random config and get reward:
    #         reward = t.clu_run(random_cfg)
    #         self.rewards[i] = (Constants.in_reward - reward) / Constants.in_reward
    #
    #         if reward < ex.best_val:
    #             ex.best_val = reward
    #             ex.best_param = random_cfg
    #             ex.best_algo = ex.clu_algos[i]
    #     # To make comparison more fair, we do not consume time for initialization
    #     # Because no actual clustering is involved, just random values
    #     # self.consume_limit(time.time() - start)

    def initialize(self, log_file):
        """
        Initialize rewards. We use here the same value,
        gained by calculating metrics on randomly assigned labels.
        """
        print("\nInit Softmax with tau = " + str(self.tau))
        self.action.data = self.action.data.withColumn('labels', round(rand()*self.params.n_clusters_upper_bound)\
                                                       .cast(IntegerType()))
        res = metric(self.action.data, self.params)
        # start = time.time()
        for i in range(0, self.params.num_algos):
            self.rewards[i] = -res  # the smallest value is, the better.
        # self.consume_limit(time.time() - start)
        log_file.write("Init rewards: " + str(self.rewards) + '\n')

    def draw(self):
        if not self.is_fair:
            s_max = self.softmax_normalize(self.rewards)
        else:
            x = np.array(self.rewards)
            x = x / (self.sum_spendings / self.n)
            s_max = self.softmax_normalize(x)

        try:
          d = choice([i for i in range(0, self.params.num_algos)], p=s_max)
        except ValueError:
          d = choice([i for i in range(0, self.params.num_algos)])
        return d

    def register_action(self, arm, time_consumed, reward):
        self.rewards[arm] += reward
        self.n[arm] += 1

    @staticmethod
    def softmax_normalize(rewards):
        x = rewards
        x = x / np.linalg.norm(x)
        e_x = np.exp(x - np.max(x))
        s_max = e_x / e_x.sum(axis=0)
        return s_max
