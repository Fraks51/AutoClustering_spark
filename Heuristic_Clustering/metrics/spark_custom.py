import numpy as np

from pyspark.sql.functions import monotonically_increasing_id, row_number
from pyspark.sql import Window
from pyspark.accumulators import AccumulatorParam


class ListAccumulatorParam(AccumulatorParam):
    def zero(self, initialValue):
        return [0] * initialValue

    def addInPlace(self, v1, v2):
        if v2 is list:
            return v1 + v2
        v1.append(v2)
        return v1


def spark_iterator(df):
    i = 0
    for row in df.toLocalIterator():
        yield i, np.array(row).astype(float)
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
