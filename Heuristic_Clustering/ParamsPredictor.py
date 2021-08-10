import sys
import time
import typing

import numpy as np
from ConfigSpace import Configuration
from ConfigSpace.util import get_one_exchange_neighbourhood
from sklearn.ensemble import RandomForestRegressor
from smac.configspace import convert_configurations_to_array
# from smac.optimizer.objective import average_cost
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost
from smac.scenario.scenario import Scenario
from smac.tae import StatusType
from smac.tae.execute_ta_run_old import ExecuteTARunOld


class ParamsPredictor(object):
    def __init__(self,
                 scenario: Scenario,
                 tae_runner: typing.Union[ExecuteTARunOld, typing.Callable],
                 expansion_number=5000,
                 batch_size=1):

        self.rng = np.random.RandomState(seed=np.random.randint(10000))
        # already in runhistory
        # aggregate_func = average_cost
        num_params = len(scenario.cs.get_hyperparameters())

        self.scenario = scenario
        self.config_space = scenario.cs
        self.runhistory = RunHistory()
        self.rh2EPM = RunHistory2EPM4Cost(scenario=scenario, num_params=num_params,
                                          success_states=[
                                              StatusType.SUCCESS,
                                              StatusType.CRASHED],
                                          impute_censored_data=False, impute_state=None)

        self.expansion_number = expansion_number
        self.batch_size = batch_size
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        # действующая конфигурация
        self.incumbent = None
        self.best_val = sys.float_info.max
        self.tae_runner = tae_runner
        self.last_turn_min = sys.float_info.max

    def get_runhistory(self):
        return self.runhistory

    def get_best_from_forest(self):
        return self.last_turn_min

    def optimize(self):

        processed = 0

        while processed < self.batch_size:
            # New configuration generation:
            # X (numpy.ndarray) – configuration vector x instance features
            # Y (numpy.ndarray) – cost values
            X, Y = self.rh2EPM.transform(self.runhistory)

            # get all found configurations sorted according to acq
            challengers = self.choose_next(X, Y)

            # Intensification:
            for ch in challengers:
                # cfg = Configuration(configuration_space=self.config_space, vector=ch)
                cfg = ch
                start_time = time.time()
                value = self.tae_runner(cfg)
                # try:
                #     debugging_printer(place='RFRS.py -> optimize\nvalue = self.tae_runner(cfg)')
                #     value = self.tae_runner(cfg)
                # except:
                #     value = sys.float_info.max

                time_spent = time.time() - start_time
                self.runhistory.add(cfg, value, time_spent, StatusType.SUCCESS)
                if value <= self.best_val:
                    self.incumbent = cfg
                    self.best_val = value
                processed += 1
                if processed >= self.batch_size:
                    break

        return self.incumbent

    def choose_next(self, configuration_vector: np.ndarray, cost_values: np.ndarray):

        if len(cost_values) != 0:
            self.model.fit(configuration_vector, cost_values.ravel())

        weighted_challengers = self.expand()

        # sort
        weighted_challengers.sort(key=lambda x: x[0])

        self.update_min(weighted_challengers[0][0])

        # drop extra
        next_configs_by_acq_value = [_[1] for _ in weighted_challengers]

        return next_configs_by_acq_value

    def expand(self) -> [Configuration]:
        configs = self.config_space.sample_configuration(size=self.expansion_number - 10)
        for i in range(len(configs)):
            configs[i].origin = 'Random Search'

        local_search_configs = self.get_by_local_search(10)
        configs.extend(local_search_configs)  # just merge lists together

        return self.sorted_by_predictions(configs)

    def sorted_by_predictions(self, configs: [Configuration]) -> [Configuration]:
        configs_arr = convert_configurations_to_array(configs)

        if self.runhistory.empty():
            predictions = np.zeros(len(configs_arr))
        else:
            predictions = self.model.predict(configs_arr)
        random_ind = self.rng.rand(len(predictions))
        # Last column is primary sort key!
        indices = np.lexsort((random_ind.flatten(), predictions.flatten()))
        # Cannot use zip here because the indices array cannot index the
        # rand_configs list, because the second is a pure python list
        return [(predictions[ind], configs[ind]) for ind in indices[::-1]]

    def update_min(self, cur_min):
        # Updates the lowest empirical cost for a configuration, across all configs

        configs = self.runhistory.get_all_configs()
        if configs:
            self.last_turn_min = min(cur_min, *[self.runhistory.get_min_cost(conf) for conf in configs])
        else:
            self.last_turn_min = cur_min

    def get_by_local_search(self, num):
        if self.runhistory.empty():
            init_points = self.config_space.sample_configuration(size=num)
        else:
            # initiate local search with best configurations from previous runs
            configs_previous_runs = self.runhistory.get_all_configs()
            configs_previous_runs_sorted = self.sorted_by_predictions(configs_previous_runs)
            init_points = list(map(lambda x: x[1], configs_previous_runs_sorted[:num]))

        configs = []
        # Start N local search from different random start points
        for start_point in init_points:
            acq_val, configuration = self._one_iter(start_point)
            configuration.origin = "Local Search"
            configs.append(configuration)

        # shuffle for random tie-break
        self.rng.shuffle(configs)

        return configs

    def _one_iter(self, start_point: Configuration) -> typing.Tuple[float, Configuration]:

        incumbent = start_point

        # Compute the acquisition value of the incumbent
        acq_val_incumbent = self.best_val

        local_search_steps = 0
        while True:
            local_search_steps += 1
            if local_search_steps > 1000:
                break

            # Get neighborhood of the current incumbent
            # by randomly drawing configurations
            changed_inc = False

            # Get one exchange neighborhood returns an iterator (in contrast of
            # the previously returned list).
            all_neighbors = get_one_exchange_neighbourhood(
                incumbent, seed=self.rng.seed())

            for neighbor in all_neighbors:
                if not neighbor.is_valid_configuration():
                    # print("WARN: invalid configuration: " + str(neighbor).replace("\n", ", "), file=sys.stderr)
                    continue
                if self.runhistory.empty():
                    acq_val = 0
                else:
                    converted = convert_configurations_to_array([neighbor])
                    acq_val = self.model.predict(converted)[0]

                if acq_val > acq_val_incumbent:
                    incumbent = neighbor
                    acq_val_incumbent = acq_val
                    changed_inc = True
                    # print("INFO: local search found cfg: " + str(neighbor).replace("\n", ", "), file=sys.stderr)
                    break

            if not changed_inc:
                break

        return acq_val_incumbent, incumbent
