import numpy as np


# Calculates the accuracy for this marker-prediction pair.
def accuracy(marker, prediction, execution_time):
    marker = marker.numpy()
    marker[marker >= 1] = 1
    prediction[prediction >= 1] = 1
    union = np.logical_or(prediction, marker)
    intersection = np.logical_and(prediction, marker)
    return intersection.sum() / union.sum()


# Returns the execution time property as a metric.
def execution_time(marker, prediction, execution_time):
    return execution_time

