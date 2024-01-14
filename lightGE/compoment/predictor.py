from lightGE.core.nn import Model
from lightGE.core.tensor import Tensor


class Predictor:
    def __iter__(self, model: Model):
        self.model = model

    def predict(self, x: Tensor):
        return self.model(x)