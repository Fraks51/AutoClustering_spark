import numpy as np
from metrics.spark_custom import *
from pyspark.accumulators import AccumulatorParam


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


def cluster_centroid(data, slack_context, n_clusters, added_columns): # data = data + labels
    rows, columns = spark_shape(data)
    centroid = [slack_context.accumulator(columns - added_columns, NumpyAccumulatorParam()) for _ in range(n_clusters)]
    num_points = [slack_context.accumulator(0) for _ in range(n_clusters)]

    def f(row, centroid, num_points):
        c = row[-(added_columns - 1)]
        centroid[c] += row[:-added_columns]
        num_points[c] += 1

    data.foreach(lambda row: f(row, centroid, num_points))
    centroid = list(map(lambda x: x.value, centroid))
    for i in range(0, n_clusters):
        centroid[i] /= num_points[i].value
    return centroid


def count_cluster_sizes(dataframe, n_clusters, spark_contexts, added_rows):
    point_in_c = [spark_contexts.accumulator(0) for _ in range(n_clusters)]

    def f(row, point_in_c):
        point_in_c[row[-(added_rows - 1)]] += 1

    dataframe.foreach(lambda row: f(row, point_in_c))
    return list(map(lambda x: x.value, point_in_c))

# not rewrite


def rotate(A, B, C):
    return (B[0]-A[0])*(C[1]-B[1])-(B[1]-A[1])*(C[0]-B[0])


def update_centroids(centroid, num_points, point, k, l, added_rows):
    for j, row_j in spark_iterator(point):
        centroid[k] *= (num_points[k] + 1)
        centroid[k] -= row_j[:-added_rows]
        if num_points[k] != 0:
            centroid[k] /= num_points[k]
        centroid[l] *= (num_points[l] - 1)
        centroid[l] += row_j[:-added_rows]
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
    split_data = data.randomSplit([1000 / size, (1 - 1000 / size)])
    row_1, row_2 = np.zeros(columns), np.zeros(columns)
    max_diam = 0
    for i, row_i in spark_iterator(split_data[0]):  # iterate elements outside cluster
        for j, row_j in spark_iterator(split_data[0]):  # iterate inside cluster
            if j >= i:
                break
            dist = euclidian_dist(row_i[:-added_column], row_j[:-added_column])
            if dist > max_diam:
                max_diam = dist
                row_1 = row_i[:-added_column]
                row_2 = row_j[:-added_column]
    acc = spark_context.accumulator({'row_1': np.array(row_1),
                                     'row_2': np.array(row_2),
                                     'dist': max_diam}
                                    , DiamAccumulatorParam())

    def f(row, acc):
        acc += np.array(row[:-added_column])

    split_data[1].foreach(lambda row: f(row, acc))
    return acc.value['dist']

