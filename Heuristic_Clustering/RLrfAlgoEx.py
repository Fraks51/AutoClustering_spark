import time

import numpy as np

from sklearn.ensemble import RandomForestRegressor

from .Constants import Constants
from .RLthreadRFRS import RLthreadRFRS
from .utils import print_log


class RLrfAlgoEx:
    # clu_algos = Constants.algos

    def __init__(self, data, metric='sil', params=None, expansion=5000):
        self.metric = metric
        self.data = data
        self.run_num = np.array([0] * params.num_algos)
        self.best_val = Constants.best_init
        self.best_param = dict()
        self.best_labels = None
        self.best_algo = ""
        self.batch_size = params.batch_size
        self.clu_algos = params.algorithms
        self.params = params
        self.same_res_counter = 0
        self.optimizers = []
        self.th = []

        # create all clustering threads in advance:
        for i in range(0, self.params.num_algos):
            self.th.append(
                RLthreadRFRS(data=self.data, algorithm_name=self.clu_algos[i], params=self.params,
                             metric=self.metric, batch_size=self.batch_size, expansion=expansion))
            self.optimizers.append(self.th[i].optimizer)

        # self.rf = RandomForestRegressor(n_estimators=1000, random_state=42)

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
            self.same_res_counter = 0
        else:
            self.same_res_counter += 1

        log_iteration = [iteration_number, self.metric, self.best_val, self.best_algo, self.clu_algos[arm],
                         reward, current_time + run_spent]
        self.params.result['iterations'] = self.params.result['iterations'] + [np.array(log_iteration)]

        print_log(file, ', '.join([str(s) for s in log_iteration])+'\n')

        file.flush()
        if self.same_res_counter >= self.params.max_no_improvement_iterations:
            return None

        # best value in random forest if the smallest one. Algo Executor provides the REWARD.
        # The smaller value is, the better reward should be.
        return -1.0 * th.optimizer.get_best_from_forest()
