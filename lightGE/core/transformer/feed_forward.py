from lightGE.core.tensor import Tensor
from lightGE.core.nn import Model, Dropout
from lightGE.core.nn import Linear
from lightGE.core.nn import ReLu


class FeedForward(Model):
    def __init__(self, n_inputs, hidden, ndrop_prob=1):
        """
        :param n_inputs:                self-attention输出的向量的长度
        :param hidden:                  FeedForward 中间隐藏层的向量长度
        """
        super(FeedForward, self).__init__()
        self.n_inputs = n_inputs
        self.hidden = hidden
        self.ndrop_prob = ndrop_prob
        self.linear1 = Linear(n_inputs, hidden)
        self.relu1 = ReLu()
        self.dropout = Dropout(p=self.ndrop_prob)
        self.linear2 = Linear(hidden, n_inputs)

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.linear1(inputs)
        x = self.relu1(x)
        x = self.dropout(x)
        return self.linear2(x)
