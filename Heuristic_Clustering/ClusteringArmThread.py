from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant

from pyspark.ml.clustering import KMeans as KMeans_spark
from pyspark.ml.clustering import GaussianMixture as GaussianMixture_spark
from pyspark.ml.clustering import BisectingKMeans as BisectingKMeans_spark
from .custom_algorithms.CustomPowerIterationClustering import CustomPowerIterationClustering

from smac.configspace import ConfigurationSpace

from Constants import Constants
import Metric


class ClusteringArmThread:

    def __init__(self, data, algorithm_name, metric, params):
        self.algorithm_name = algorithm_name
        self.metric = metric
        self.data = data
        self.current_labels = None
        self.n_clusters_upper_bound = params.n_clusters_upper_bound
        self.value = Constants.bad_cluster
        self.parameters = dict()
        self.configuration_space = ConfigurationSpace()

        if algorithm_name == Constants.kmeans_algo:
            self.configuration_space.add_hyperparameters(self.get_kmeans_configspace(self.n_clusters_upper_bound))

        elif algorithm_name == Constants.gm_algo:
            self.configuration_space.add_hyperparameters(
                self.get_gaussian_mixture_configspace(self.n_clusters_upper_bound))

        elif algorithm_name == Constants.bisecting_kmeans:
            self.configuration_space.add_hyperparameters(
                self.get_bisecting_kmeans_configspace(self.n_clusters_upper_bound))

        elif algorithm_name == Constants.fully_connected_pic:
            self.configuration_space.add_hyperparameters(self.get_pic_configspace(self.n_clusters_upper_bound))

    def update_labels(self, configuration):
        if self.algorithm_name == Constants.kmeans_algo:
            algorithm = KMeans_spark(predictionCol='labels', **configuration)
        elif self.algorithm_name == Constants.gm_algo:
            algorithm = GaussianMixture_spark(predictionCol='labels', **configuration)
        elif self.algorithm_name == Constants.bisecting_kmeans:
            algorithm = BisectingKMeans_spark(predictionCol='labels', **configuration)
        elif self.algorithm_name == Constants.fully_connected_pic:
            algorithm = CustomPowerIterationClustering(predictionCol='labels', **configuration)

        if self.algorithm_name == Constants.fully_connected_pic:
            self.current_labels = algorithm.fit_transform(self.data)
        else:
            model = algorithm.fit(self.data)
            predictions = model.transform(self.data)
            self.current_labels = predictions

    def clu_run(self, cfg):
        self.update_labels(cfg)
        return Metric.metric(self.current_labels)

    @staticmethod
    def get_kmeans_configspace(n_clusters_upper_bound):
        """
        k : number of clusters
        initMode : The initialization algorithm. This can be either "random" to choose random points as initial cluster
                   centers, or "k-means||" to use a parallel variant of k-means++
        initSteps : The number of steps for k-means|| initialization mode. Must be > 0
        maxIter : max number of iterations (>= 0)
        seed : random seed
        distanceMeasure : Supported options: 'euclidean' and 'cosine'.

        Returns
        -----------------
        Tuple of parameters
        """
        k = UniformIntegerHyperparameter("k", 2, n_clusters_upper_bound)
        initMode = CategoricalHyperparameter("initMode", ['random', 'k-means||'])
        initSteps = UniformIntegerHyperparameter("initSteps", 1, 5)
        maxIter = UniformIntegerHyperparameter("maxIter", 5, 50)
        distanceMeasure = CategoricalHyperparameter("distanceMeasure", ['euclidean', 'cosine'])
        return k, initMode, initSteps, maxIter, distanceMeasure

    @staticmethod
    def get_gaussian_mixture_configspace(n_clusters_upper_bound):
        """
        k : number of clusters
        aggregationDepth : suggested depth for treeAggregate (>= 2)
        maxIter : max number of iterations (>= 0)
        tol : the convergence tolerance for iterative algorithms (>= 0)

        Returns
        -------
        Tuple of parameters
        """
        k = UniformIntegerHyperparameter("k", 2, n_clusters_upper_bound)
        maxIter = UniformIntegerHyperparameter("maxIter", 5, 50)
        tol = UniformFloatHyperparameter("tol", 1e-6, 0.1)
        aggregationDepth = UniformIntegerHyperparameter("aggregationDepth", 2, 15)
        return k, maxIter, tol, aggregationDepth

    @staticmethod
    def get_bisecting_kmeans_configspace(n_clusters_upper_bound):
        """
        k : number of clusters
        initSteps : The number of steps for k-means|| initialization mode. Must be > 0
        maxIter : max number of iterations (>= 0)
        seed : random seed
        distanceMeasure : Supported options: 'euclidean' and 'cosine'.
        minDivisibleClusterSize : The minimum number of points (if >= 1.0) or the minimum proportion of points
                                                               (if < 1.0) of a divisible cluster.
                                                               we use only proportion
        Returns
        -----------------
        Tuple of parameters
        """
        k = UniformIntegerHyperparameter("k", 2, n_clusters_upper_bound)
        maxIter = UniformIntegerHyperparameter("maxIter", 5, 50)
        distanceMeasure = CategoricalHyperparameter("distanceMeasure", ['euclidean', 'cosine'])
        minDivisibleClusterSize = UniformFloatHyperparameter("minDivisibleClusterSize", 0.01, 1.0)
        return k, maxIter, distanceMeasure, minDivisibleClusterSize

    @staticmethod
    def get_pic_configspace(n_clusters_upper_bound):
        """
        k : number of clusters
        maxIter : max number of iterations (>= 0)
        initMode : The initialization algorithm. This can be either 'random' to use a random vector as vertex
        properties, or 'degree' to use a normalized sum of similarities with other vertices.
        Supported options: 'random' and 'degree'.

        Returns
        -----------------
        Tuple of parameters
        """
        k = UniformIntegerHyperparameter("k", 2, n_clusters_upper_bound)
        maxIter = UniformIntegerHyperparameter("maxIter", 5, 50)
        initMode = CategoricalHyperparameter("initMode", ['random', 'degree'])