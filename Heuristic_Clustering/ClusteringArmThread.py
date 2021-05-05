import threading
import traceback

import numpy as np
import sys
from ConfigSpace.conditions import InCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant

from pyspark.ml.clustering import KMeans as KMeans_spark
from pyspark.ml.clustering import GaussianMixture as GaussianMixture_spark
from pyspark.ml.clustering import BisectingKMeans as BisectingKMeans_spark

from smac.configspace import ConfigurationSpace
from pyspark.ml.feature import VectorAssembler

import Constants
import Metric


class ClusteringArmThread:

    def __init__(self, data, algorithm_name, metric, seed):
        self.algorithm_name = algorithm_name
        self.metric = metric
        if algorithm_name in Constants.rewrited:
            self.data = data
            vector_assembler = VectorAssembler(inputCols=self.data.columns,
                                               outputCol="features")
            self.data = vector_assembler.transform(self.data)
        else:
            self.data = data.toPandas().values
        self.value = Constants.bad_cluster
        self.parameters = dict()
        self.seed = seed
        self.configuration_space = ConfigurationSpace()

        if algorithm_name == Constants.kmeans_algo:
            self.configuration_space.add_hyperparameters(self.get_kmeans_configspace())

        elif algorithm_name == Constants.gm_algo:
            self.configuration_space.add_hyperparameters(self.get_gaussian_mixture_configspace())

        elif algorithm_name == Constants.bisecting_kmeans:
            self.configuration_space.add_hyperparameters(self.get_bisecting_kmeans_configspace())

    def get_labels(self, configuration):
        model = None
        if self.algorithm_name == Constants.kmeans_algo:
            model = KMeans_spark(predictionCol='labels', **configuration)
        elif self.algorithm_name == Constants.gm_algo:
            model = GaussianMixture_spark(predictionCol='labels', **configuration)
        elif self.algorithm_name == Constants.bisecting_kmeans:
            model = BisectingKMeans_spark(predictionCol='labels', **configuration)

        if Constants.DEBUG:
            model.fit(self.data)
        else:
            # Some problems with smac, old realization, don't change
            try:
                model.fit(self.data)
            except:
                try:
                    exc_info = sys.exc_info()
                    try:
                        model.fit(self.data)  # try again
                    except:
                        pass
                finally:
                    print("Error occured while fitting " + self.algorithm_name)
                    print("Error occured while fitting " + self.algorithm_name, file=sys.stderr)
                    traceback.print_exception(*exc_info)
                    del exc_info
                    return Constants.bad_cluster

        if self.algorithm_name in Constants.rewrited:
            predictions = model.transform(self.data)
            labels = predictions
        # elif (self.algorithm_name == Constants.gm_algo) or (self.algorithm_name == Constants.bgm_algo):
        #     labels = model.predict(self.data)
        else:
            labels = model.labels_

        return labels

    def clu_run(self, cfg):
        labels = self.get_labels(cfg)
        # labels_unique = np.unique(labels)
        # n_clusters = len(labels_unique)
        # value = Metric.metric(self.data, n_clusters, labels, self.metric)
        # return value
        return Metric.metric(self.data)

    @staticmethod
    def get_kmeans_configspace():
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
        k = UniformIntegerHyperparameter("k", 2, Constants.n_clusters_upper_bound)
        initMode = CategoricalHyperparameter("initMode", ['random', 'k-means||'])
        initSteps = UniformIntegerHyperparameter("initSteps", 1, 5)
        maxIter = UniformIntegerHyperparameter("maxIter", 5, 50)
        distanceMeasure = CategoricalHyperparameter("distanceMeasure", ['euclidean', 'cosine'])
        return k, initMode, initSteps, maxIter, distanceMeasure

    @staticmethod
    def get_gaussian_mixture_configspace():
        """
        k : number of clusters
        aggregationDepth : suggested depth for treeAggregate (>= 2)
        maxIter : max number of iterations (>= 0)
        tol : the convergence tolerance for iterative algorithms (>= 0)

        Returns
        -------
        Tuple of parameters
        """
        k = UniformIntegerHyperparameter("k", 2, Constants.n_clusters_upper_bound)
        maxIter = UniformIntegerHyperparameter("maxIter", 5, 50)
        tol = UniformFloatHyperparameter("tol", 1e-6, 0.1)
        return k, maxIter, tol

    @staticmethod
    def get_bisecting_kmeans_configspace():
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
        k = UniformIntegerHyperparameter("k", 2, Constants.n_clusters_upper_bound)
        initSteps = UniformIntegerHyperparameter("initSteps", 1, 5)
        maxIter = UniformIntegerHyperparameter("maxIter", 5, 50)
        distanceMeasure = CategoricalHyperparameter("distanceMeasure", ['euclidean', 'cosine'])
        minDivisibleClusterSize = UniformFloatHyperparameter("minDivisibleClusterSize", 0.01, 1.0)
        return k, initSteps, maxIter, distanceMeasure, minDivisibleClusterSize