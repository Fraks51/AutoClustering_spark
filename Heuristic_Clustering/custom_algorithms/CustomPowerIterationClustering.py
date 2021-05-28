from pyspark.sql.functions import udf, col
from pyspark.sql.types import *
from pyspark.ml.feature import Interaction, VectorAssembler
from pyspark.sql.functions import desc, row_number, monotonically_increasing_id
from pyspark.sql.window import Window
from scipy.spatial import distance
from pyspark.ml.clustering import PowerIterationClustering


class CustomPowerIterationClustering:
    def __init__(self, predictionCol='labels', dist_func=distance.euclidean, **configuration):
        self.algorithm = PowerIterationClustering(**configuration)
        self.predictionCol = predictionCol
        self.dist_func = dist_func

    def fit_transform(self, data):
        udffunc = udf(self.dist_func, FloatType())
        indexed_data = data.select('features') \
            .withColumn('src', row_number().over(Window.orderBy(monotonically_increasing_id())))
        res = indexed_data.join(indexed_data.toDF('features2', 'dst')) \
            .filter("src < dst") \
            .withColumn("weight", udffunc("features", "features2"))

        clustered_df = self.algorithm.assignClusters(res).withColumnRenamed('cluster', 'labels')
        return clustered_df.join(indexed_data, clustered_df.id == indexed_data.src)
