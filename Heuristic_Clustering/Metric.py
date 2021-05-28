from py4j.protocol import Py4JJavaError
from pyspark.ml.evaluation import ClusteringEvaluator
import sys

# TODO: change when more metrics arrived
# TODO: delete prints when found where use metrics
def metric(data, **kwargs):
    try:
        res = -ClusteringEvaluator(predictionCol='labels', distanceMeasure='squaredEuclidean').evaluate(data)
        return res
    except TypeError:
        print("\n\nTYPE ERROR OCCURED IN Metric.py:\n\nDATA: {}\n\n".format(data))
        return 0
    except Py4JJavaError:
        print("\n\nPy4JJavaError ERROR OCCURED IN Metric.py:\n\nDATA: {}\n\n".format(data.printSchema()))
        return sys.float_info.max
