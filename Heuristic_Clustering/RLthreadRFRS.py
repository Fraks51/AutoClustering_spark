from pyspark.sql.dataframe import DataFrame

from ClusteringArmThread import ClusteringArmThread
from ParamsPredictor import ParamsPredictor
from smac.scenario.scenario import Scenario


class RLthreadRFRS(ClusteringArmThread):
    def __init__(self, data: DataFrame, algorithm_name: str, metric: str, batch_size: int,
                 expansion=5000, params=None):
        self.run_count = batch_size
        ClusteringArmThread.__init__(self, data, algorithm_name, metric,
                                     params=params)  # populates config space
        self.new_scenario(1)  # initial scenario

        self.optimizer = ParamsPredictor(scenario=self.clu_scenario,
                                         tae_runner=self.clu_run,
                                         expansion_number=expansion,
                                         batch_size=batch_size)

    def new_scenario(self, c, remaining_time=None):
        # remaining_time is usually expected to be way more than needed for one call,
        # but sometimes it's guarding from hanging arbitraty long in single iteration
        if remaining_time is None:
            self.clu_scenario = Scenario({"run_obj": "quality",
                                          "cs": self.configuration_space,
                                          "deterministic": "true",
                                          "runcount-limit": self.run_count * c
                                          })
        else:
            self.clu_scenario = Scenario({"run_obj": "quality",
                                          "cs": self.configuration_space,
                                          "deterministic": "true",
                                          "tuner-timeout": remaining_time,
                                          "wallclock_limit": remaining_time,
                                          "cutoff_time": remaining_time,
                                          "runcount-limit": self.run_count * c
                                          })

    def run(self):
        print('Run RFRS ' + self.algorithm_name)
        self.parameters = self.optimizer.optimize()
        self.value = self.optimizer.get_runhistory().get_cost(self.parameters)
