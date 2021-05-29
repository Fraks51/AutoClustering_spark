from Constants import Constants


class Parameters:
    def __init__(self, spark_context, algorithms=Constants.algos, n_clusters_upper_bound=15,
                 bandit_timeout=30, time_limit=1000, batch_size=40, MAB_solver='ucb', tau=0.5,
                 max_no_improvement_iterations=None, metric='sil'):
        self.result = dict()
        self.result['iterations'] = []
        self.spark_context = spark_context
        self.algorithms = algorithms
        self.num_algos = len(self.algorithms)
        self.n_clusters_upper_bound = n_clusters_upper_bound
        # seconds for each bandit iteration
        self.bandit_timeout = bandit_timeout
        # iterations number
        # self.bandit_iterations = bandit_iterations
        self.batch_size = batch_size
        # self.tuner_timeout = bandit_timeout * (bandit_iterations + 1) / self.num_algos
        self.tuner_timeout = time_limit
        self.MAB_solver = MAB_solver
        self.tau = tau
        self.metric = metric
        if max_no_improvement_iterations is None:
            self.max_no_improvement_iterations = float('inf')
        else:
            self.max_no_improvement_iterations = max_no_improvement_iterations
