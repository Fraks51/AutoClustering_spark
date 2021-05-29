import numpy as np
import math

from pyspark.sql.functions import mean as _mean, col
from metrics.utils import *
from metrics.measure import Measure
from pyspark import SparkContext, SQLContext


class DaviesIndex(Measure):

    def __init__(self):
        self.s_clusters = []
        self.cluster_sizes = []
        self.centroids = []
        self.sums = []
        self.diameter = 0

    def s(self, X, cluster_k_index, cluster_sizes, centroids, spark_context):
        acc = spark_context.accumulator(0.0)

        def f(row, acc, centroids):
            if row[-2] == cluster_k_index:
                acc += euclidian_dist(row[:-3], centroids[cluster_k_index])

        X.foreach(lambda row: f(row, acc, centroids))

        if cluster_sizes[cluster_k_index] == 0:
            return float('inf')
        return acc.value / cluster_sizes[cluster_k_index]

    def find(self, data, spark_context):
        n_clusters = get_n_clusters(data, data.columns[-1])
        self.diameter = find_diameter(data, spark_context, 2)
        self.s_clusters = [0. for _ in range(n_clusters)]
        self.centroids = cluster_centroid(data, spark_context, n_clusters, 2)
        db = 0
        self.cluster_sizes = count_cluster_sizes(data, n_clusters, spark_context, 2)
        cluster_dists = [spark_context.accumulator(0.0) for _ in range(n_clusters)]

        def f(row, cluster_dists, centroids):
            label = row[-1]
            cluster_dists[label] += euclidian_dist(row[:-2], centroids[label])

        data.foreach(lambda row: f(row, cluster_dists, self.centroids))

        for i in range(n_clusters):
            if self.cluster_sizes[i] == 0:
                self.s_clusters[i] = float('inf')
            else:
                self.s_clusters[i] = cluster_dists[i].value / self.cluster_sizes[i]
        self.sums = [[0 for _ in range(n_clusters)] for _ in range(n_clusters)]
        for i in range(0, n_clusters):
            for j in range(0, n_clusters):
                if i != j:
                    tm = euclidian_dist(self.centroids[i], self.centroids[j])
                    if tm != 0:
                        self.sums[i][j] = (self.s_clusters[i] + self.s_clusters[j]) / tm
                    else:
                        pass
                        #a = -Constants.bad_cluster
            tmp = np.amax(self.sums[i])
            db += tmp
        db /= float(n_clusters)
        return db

    def update(self, data, spark_context, n_clusters, k, l, ids):
        rows, columns = spark_shape(data)
        columns -= 2
        sql = SQLContext(spark_context)
        label_name = data.columns[-1]
        df = add_iter(data)
        delta = 10 ** (-math.log(rows, 10) - 1)
        points = df.filter(df.row_idx.isin(ids))
        prev_centroids = np.copy(self.centroids)
        # self.cluster_sizes = cluster_centroid.count_cluster_sizes(labels, n_clusters)
        self.centroids = update_centroids(self.centroids, np.copy(self.cluster_sizes), points, k, l, 3)

        if euclidian_dist(prev_centroids[k], self.centroids[k]) > delta * self.diameter:
            self.s_clusters[k] = self.s(df, k, self.cluster_sizes, self.centroids, spark_context)

        if euclidian_dist(prev_centroids[l], self.centroids[l]) > delta * self.diameter:
            self.s_clusters[l] = self.s(df, l, self.cluster_sizes, self.centroids, spark_context)

        db = 0
        for i in range(n_clusters):
            if i != k:
                tm = euclidian_dist(self.centroids[i], self.centroids[k])
                if tm != 0:
                    self.sums[i][k] = (self.s_clusters[i] + self.s_clusters[k]) / tm
                    self.sums[k][i] = (self.s_clusters[i] + self.s_clusters[k]) / tm
            if i != l:
                tm = euclidian_dist(self.centroids[i], self.centroids[l])
                if tm != 0:
                    self.sums[i][l] = (self.s_clusters[i] + self.s_clusters[l]) / tm
                    self.sums[l][i] = (self.s_clusters[i] + self.s_clusters[l]) / tm
        for i in range(n_clusters):
            tmp = np.amax(self.sums[i])
            db += tmp
        db /= float(n_clusters)
        return db

