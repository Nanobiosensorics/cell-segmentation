from cellpose import models
from src.methods.method import Method, add_execution_time


class BasicCellposeMethod(Method):
    def __init__(self, gpu=False):
        self.model = models.Cellpose(gpu, model_type='cyto3')

    @add_execution_time
    def predict(self, im_mic):
        masks, _, _, _ = self.model.eval(im_mic, diameter=None, channels=[0, 0])
        return masks

