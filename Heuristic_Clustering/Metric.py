from py4j.protocol import Py4JJavaError
from pyspark.ml.evaluation import ClusteringEvaluator
# from metrics.ch_index import ChIndex
import sys
import numpy as np
import math

from pyspark.sql.functions import monotonically_increasing_id, row_number
from pyspark.sql.functions import mean as _mean, col
from pyspark import SparkContext, SQLContext
from pyspark.sql import Window
from pyspark.accumulators import AccumulatorParam
from abc import abstractmethod, ABC


class Measure(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def find(self, data, spark_context):
        pass

    @abstractmethod
    def update(self, data, spark_context, n_clusters, k, l, id):
        '''
        :param k: old_labels for id
        :param l: new_labels for id
        :param id: ids for rows with new labels
        '''
        pass


# Spark custom

class ListAccumulatorParam(AccumulatorParam):
    def zero(self, initialValue):
        return [0] * initialValue

    def addInPlace(self, v1, v2):
        if v2 is list:
            return v1 + v2
        v1.append(v2)
        return v1


def spark_iterator(df, added_column):
    i = 0
    for row in df.toLocalIterator():
        yield i, np.array(row[:-added_column]).astype(float)
        i += 1


def get_n_clusters(df, label_column):
    return df.groupBy(label_column).count().count()


def add_iter(df):
    df = df.withColumn("row_idx", row_number().over(Window.orderBy(monotonically_increasing_id())))
    return df


def spark_join(df, column, column_name, sql_context):
    b = sql_context.createDataFrame([(int(l),) for l in column], [column_name])

    # add 'sequential' index and join both dataframe to get the final result
    df = df.withColumn("row_idx_2", row_number().over(Window.orderBy(monotonically_increasing_id())))
    b = b.withColumn("row_idx", row_number().over(Window.orderBy(monotonically_increasing_id())))

    final_df = df.join(b, df.row_idx_2 == b.row_idx). \
        drop("row_idx_2")
    return final_df


# Utils

def euclidian_dist(x1, x2):
    return np.sqrt(np.sum(np.square(x1 - x2)))


def none_check(param):
    if param is None:
        param = []
    return param


def spark_shape(df):
    return df.count(), len(df.columns)


class NumpyAccumulatorParam(AccumulatorParam):
    def zero(self, initialValue):
        return np.zeros(initialValue)

    def addInPlace(self, v1, v2):
        v1 += v2
        return v1


def cluster_centroid(data, slack_context, n_clusters, added_columns):  # data = data + labels
    rows, columns = spark_shape(data)
    centroid = [slack_context.accumulator(columns - added_columns, NumpyAccumulatorParam()) for _ in range(n_clusters)]
    num_points = [slack_context.accumulator(0) for _ in range(n_clusters)]

    def f(row, centroid, num_points):
        c = row[-(added_columns - 1)]
        centroid[c] += row[:-added_columns]
        num_points[c] += 1

    data.rdd.foreach(lambda row: f(row, centroid, num_points))
    centroid = list(map(lambda x: x.value, centroid))
    for i in range(0, n_clusters):
        centroid[i] /= num_points[i].value
    return centroid


def count_cluster_sizes(dataframe, n_clusters, spark_contexts, added_rows):
    point_in_c = [spark_contexts.accumulator(0) for _ in range(n_clusters)]

    def f(row, point_in_c):
        point_in_c[row[-(added_rows - 1)]] += 1

    dataframe.rdd.foreach(lambda row: f(row, point_in_c))
    return list(map(lambda x: x.value, point_in_c))


# not rewrite


def rotate(A, B, C):
    return (B[0] - A[0]) * (C[1] - B[1]) - (B[1] - A[1]) * (C[0] - B[0])


def update_centroids(centroid, num_points, point, k, l, added_rows):
    for j, row_j in spark_iterator(point, added_rows):
        centroid[k] *= (num_points[k] + 1)
        centroid[k] -= row_j
        if num_points[k] != 0:
            centroid[k] /= num_points[k]
        centroid[l] *= (num_points[l] - 1)
        centroid[l] += row_j
        centroid[l] /= num_points[l]
    return centroid


class DiamAccumulatorParam(AccumulatorParam):
    def zero(self, value):
        return value

    def addInPlace(self, v1, v2):
        if type(v2) is dict:
            dist = []
            dist.append(v1['dist'])
            dist.append(v2['dist'])
            dist.append(euclidian_dist(v1['row_1'], v2['row_1']))
            dist.append(euclidian_dist(v1['row_1'], v2['row_2']))
            dist.append(euclidian_dist(v1['row_2'], v2['row_1']))
            dist.append(euclidian_dist(v1['row_2'], v2['row_2']))
            i = np.argmax(dist)
            v1['dist'] = dist[i]
            if i == 1:
                v1['row_1'] = v2['row_1']
                v1['row_2'] = v2['row_2']
            elif i == 2:
                v1['row_2'] = v2['row_1']
            elif i == 3:
                v1['row_2'] = v2['row_2']
            elif i == 4:
                v1['row_1'] = v2['row_1']
            elif i == 5:
                v1['row_1'] = v2['row_2']
            return v1
        dist_1 = euclidian_dist(v1['row_1'], v2)
        dist_2 = euclidian_dist(v2, v1['row_2'])
        if dist_1 >= v1['dist'] and dist_1 >= dist_2:
            v1['dist'] = dist_1
            v1['row_2'] = v2
        elif dist_2 >= v1['dist'] and dist_2 >= dist_1:
            v1['dist'] = dist_2
            v1['row_1'] = v2
        return v1


def find_diameter(data, spark_context, added_column):
    size, columns = spark_shape(data)
    columns -= added_column
    if size <= 1000:
        split_data = data.randomSplit([0.25, 0.75])
    else:
        split_data = data.randomSplit([1000 / size, (1 - 1000 / size)])
    row_1, row_2 = np.zeros(columns), np.zeros(columns)
    max_diam = 0
    for i, row_i in spark_iterator(split_data[0], added_column):  # iterate elements outside cluster
        for j, row_j in spark_iterator(split_data[0], added_column):  # iterate inside cluster
            if j >= i:
                break
            dist = euclidian_dist(row_i, row_j)
            if dist > max_diam:
                max_diam = dist
                row_1 = row_i
                row_2 = row_j
    acc = spark_context.accumulator({'row_1': np.array(row_1),
                                     'row_2': np.array(row_2),
                                     'dist': max_diam}
                                    , DiamAccumulatorParam())

    def f(row, acc):
        acc += np.array(row[:-added_column])

    split_data[1].rdd.foreach(lambda row: f(row, acc))
    return acc.value['dist']


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

    @staticmethod
    def f(row, acc, centroid):
        acc += 1

    def find(self, data, spark_context):
        rows, columns = spark_shape(data)
        n_clusters = get_n_clusters(data, data.columns[-1])
        columns -= 2
        mean_columns = map(lambda x: _mean(col(x)).alias('mean'), data.columns[:-2])
        df_stats = data.select(
            *mean_columns
        ).collect()
        df = add_iter(data)
        self.x_center = np.array(df_stats[0])
        self.centroids = cluster_centroid(df, spark_context, n_clusters, 3)
        self.diameter = find_diameter(df, spark_context, 3)
        ch = float(rows - n_clusters) / float(n_clusters - 1)

        self.cluster_sizes = count_cluster_sizes(df, n_clusters, spark_context, 3)
        self.numerator = [0 for _ in range(n_clusters)]
        for i in range(0, n_clusters):
            self.numerator[i] = self.cluster_sizes[i] * euclidian_dist(self.centroids[i], self.x_center)
        denominator_sum = spark_context.accumulator(0)

        # def f(row, acc, centroid):
        #     acc += euclidian_dist(row[:-3], centroid[row[-2]])
        # df.rdd.foreach(lambda row: f(row, denominator_sum, self.centroids))

        def f(row, denominator_sum, centroind):
            denominator_sum += np.sqrt(np.sum(
                np.square(np.array(row[:-3]) - centroind[row[-2]])))

        centroind = self.centroids

        df.rdd.foreach(lambda row: f(row, denominator_sum, centroind))

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

        df.rdd.foreach(lambda row: f(k, l, prev_centroids, self.centroids, delta, self.diameter, row, denom))
        self.denominator += denom.value
        ch *= sum(self.numerator)
        ch /= self.denominator
        return -ch


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

        X.rdd.foreach(lambda row: f(row, acc, centroids))

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

        data.rdd.foreach(lambda row: f(row, cluster_dists, self.centroids))

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
                        # a = -Constants.bad_cluster
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


def metric(data, params):
    print(params.metric)
    try:
        if params.metric == 'sil':
            res = -ClusteringEvaluator(predictionCol='labels', distanceMeasure='squaredEuclidean').evaluate(data)
        elif params.metric == 'ch':
            res = ChIndex().find(data, params.spark_context)
        elif params.metric == 'db':
            res = DaviesIndex().find(data, params.spark_context)
        return res
    except TypeError:
        print("\n\nTYPE ERROR OCCURED IN Metric.py:\n\nDATA: {}\n\n".format(data))
        return 0
    # except Py4JJavaError:
    #     print("\n\nPy4JJavaError ERROR OCCURED IN Metric.py:\n\nDATA: {}\n\n".format(data.printSchema()))
    #     return sys.float_info.max
