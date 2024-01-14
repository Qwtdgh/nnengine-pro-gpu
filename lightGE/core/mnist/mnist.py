
from lightGE.core import Sequential, Conv2d, Model, MaxPool2d, ReLu, Dropout2d, Linear, Tensor


class MNIST(Model):
    def __init__(self):
        super(MNIST, self).__init__()

        self.conv1 = Sequential(
            [Conv2d(1, 10, filter_size=5),
             MaxPool2d(filter_size=2),
             ReLu()])

        self.conv2 = Sequential([
            Conv2d(10, 20, filter_size=5),
            Dropout2d(),
            MaxPool2d(filter_size=2),
            ReLu()])
        self.fc1 = Sequential([Linear(320, 50), ReLu()])
        self.fc2 = Sequential([Dropout2d(), Linear(50, 10)])

    def forward(self, x: Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape((x.shape[0], -1))
        x = self.fc1(x)
        x = self.fc2(x)
        return x.softmax().log()

    def predict(self, input: Tensor):
        return self.forward(input)
