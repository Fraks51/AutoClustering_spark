import numpy as np
import math

from pyspark.sql.functions import mean as _mean, col
from metrics.utils import *
from metrics.measure import Measure
from pyspark import SparkContext, SQLContext


class ChIndex(Measure):
    '''
    Note: doesn't found info about ch-index
    '''

    def __init__(self, centroids=None, cluster_sizes=None, x_center=0, numerator=None,
                 denominator=None, diameter=0):
        self.centroids = none_check(centroids)
        self.cluster_sizes = none_check(cluster_sizes)
        self.x_center = x_center
        self.numerator = none_check(numerator)
        self.denominator = none_check(denominator)
        self.diameter = diameter

    def find(self, data, spark_context):
        rows, columns = spark_shape(data)
        n_clusters = get_n_clusters(data, data.columns[-1])
        columns -= 2
        mean_columns = map(lambda x: _mean(col(x)).alias('mean'), data.columns)
        df_stats = data.select(
            *mean_columns
        ).collect()
        df = add_iter(data)
        self.x_center = np.array(df_stats[0][:-2])
        self.centroids = cluster_centroid(df, spark_context, n_clusters, 3)
        self.diameter = find_diameter(df, spark_context, 3)
        ch = float(rows - n_clusters) / float(n_clusters - 1)

        self.cluster_sizes = count_cluster_sizes(df, n_clusters, spark_context, 3)
        self.numerator = [0 for _ in range(n_clusters)]
        for i in range(0, n_clusters):
             self.numerator[i] = self.cluster_sizes[i] * euclidian_dist(self.centroids[i], self.x_center)
        denominator_sum = spark_context.accumulator(0)

        def f(row, acc, centroid):
            acc += euclidian_dist(row[:-3], centroid[row[-2]])

        df.foreach(lambda row: f(row, denominator_sum, self.centroids))
        self.denominator = denominator_sum.value
        ch *= np.sum(self.numerator)
        ch /= self.denominator
        return -ch

    def update(self, data, spark_context, n_clusters, k, l, id):
        rows, columns = spark_shape(data)
        columns -= 2
        sql = SQLContext(spark_context)
        label_name = data.columns[-1]
        df = add_iter(data)
        delta = 10 ** (-math.log(rows, 10) - 1)
        point = df.filter(df.row_idx.isin(id))
        prev_centroids = np.copy(self.centroids)
        self.cluster_sizes = count_cluster_sizes(df, n_clusters, spark_context, 3)
        self.centroids = update_centroids(self.centroids, np.copy(self.cluster_sizes), point, k, l, 3)
        ch = float(rows - n_clusters) / float(n_clusters - 1)
        self.numerator[k] = self.cluster_sizes[k] * euclidian_dist(self.centroids[k], self.x_center)
        self.numerator[l] = self.cluster_sizes[l] * euclidian_dist(self.centroids[l], self.x_center)
        denom = spark_context.accumulator(0)

        def f(k, l, prev_centroids, centroids, delta, diameter, row, denom):
            if (row[-2] == k and euclidian_dist(prev_centroids[k], centroids[k]) > delta * diameter
               or row[-2] == l and euclidian_dist(prev_centroids[l], centroids[l]) > delta * diameter):
                denom += (euclidian_dist(row[:-3], centroids[row[-2]])
                                     - euclidian_dist(row[:-3], prev_centroids[row[-2]]))

        df.foreach(lambda row: f(k, l, prev_centroids, self.centroids, delta, self.diameter, row, denom))
        self.denominator += denom.value
        ch *= sum(self.numerator)
        ch /= self.denominator
        return -ch
