import timeit
from abc import ABC, abstractmethod


# Decorator for adding execution time property when making a prediction.
def add_execution_time(predict):
    def func(self, im_mic):
        start_time = timeit.default_timer()
        result = predict(self, im_mic)
        elapsed = timeit.default_timer() - start_time
        return result, elapsed
    return func


# Base class of an prediction method.
class Method(ABC):
    # Returns the markers and the execution time.
    @abstractmethod
    def predict(self, im_mic):
        pass

    # Default batch prediction, for pytorch it's better to override it.
    @add_execution_time
    def predict_batch(self, im_batch):
        result = []
        for img in im_batch:
            markers, _ = self.predict(img)
            result.append(markers)
        return result

