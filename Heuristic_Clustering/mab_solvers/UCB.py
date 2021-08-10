import math

import numpy as np
from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType

import Metric
from mab_solvers.MabSolver import MabSolver


class UCB(MabSolver):
    def __init__(self, action, is_fair=False, params=None):
        MabSolver.__init__(self, action, params)
        self.num_algos = params.num_algos
        self.rewards = np.zeros(params.num_algos)
        self.n = np.array([1] * self.num_algos)
        self.name = "ucb"
        self.iter = 1
        self.is_fair = is_fair

    def initialize(self, log_file):
        """
        Initialize rewards. We use here the same value,
        gained by calculating metrics on randomly assigned labels.
        """

        # Random initialization of cluster labels
        self.action.data = self.action.data.withColumn('labels', round(rand()*self.params.n_clusters_upper_bound)\
                                                       .cast(IntegerType()))

        res = Metric.metric(self.action.data, self.params.metric)

        for i in range(0, self.params.num_algos):
            self.rewards[i] = -res  # the smallest value is, the better.
        log_file.write("Init rewards: " + str(self.rewards) + '\n')

    def draw(self):
        values = self.rewards
        if self.is_fair:
            values = values / (self.sum_spendings / self.n)

        values = values + math.sqrt(2 * math.log(self.iter)) / self.n
        return np.argmax(values)

    def register_action(self, arm, time_consumed, reward):
        self.iter += 1
        self.rewards[arm] += reward
        self.n[arm] += 1
