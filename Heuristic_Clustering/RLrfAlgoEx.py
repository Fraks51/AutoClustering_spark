import time

import numpy as np

from sklearn.ensemble import RandomForestRegressor

import Constants
from RLthreadRFRS import RLthreadRFRS
from utils import debugging_printer


class RLrfAlgoEx:
    clu_algos = Constants.algos

    def __init__(self, data, metric='sil', seed=42, batch_size=Constants.batch_size, expansion=5000):
        self.metric = metric
        self.data = data
        self.run_num = np.array([0] * Constants.num_algos)
        self.best_val = Constants.best_init
        self.best_param = dict()
        self.best_labels = None
        self.best_algo = ""
        self.seed = seed
        self.batch_size = batch_size
        self.optimizers = []
        self.th = []

        # create all clustering threads in advance:
        for i in range(0, Constants.num_algos):
            self.th.append(
                RLthreadRFRS(data=self.data, algorithm_name=self.clu_algos[i],\
                             metric=self.metric, seed=self.seed, batch_size=self.batch_size, expansion=expansion))
            self.optimizers.append(self.th[i].optimizer)

        self.rf = RandomForestRegressor(n_estimators=1000, random_state=42)

    def apply(self, arm, file, iteration_number, remaining_time=None, current_time=0):
        # RLthreadRFRS[arm]
        th = self.th[arm]

        # initially, run_num for each arm == 0, thus we allocate 1 batch of target f calls:
        th.new_scenario(c=self.run_num[arm] + 1, remaining_time=remaining_time)  # add budget

        run_start = time.time()
        th.run()
        run_spent = int(time.time() - run_start)

        self.run_num[arm] += 1
        reward = th.value

        if reward < self.best_val:
            self.best_val = reward
            self.best_param = th.parameters
            self.best_algo = th.algorithm_name
            self.best_labels = th.current_labels
        log_string = str(iteration_number) \
                     + ', ' + self.metric \
                     + ', ' + str(self.best_val) \
                     + ', ' + str(self.best_algo) \
                     + ', ' + str(self.clu_algos[arm]) \
                     + ', ' + str(reward) \
                     + ', ' + str(current_time + run_spent)

        file.write(log_string + '\n')
        file.flush()

        # best value in random forest if the smallest one. Algo Executor provides the REWARD.
        # The smaller value is, the better reward should be.
        return -1.0 * th.optimizer.get_best_from_forest()
