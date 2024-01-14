import tqdm

from lightGE.data.dataloader import Dataset
from lightGE.core import Tensor, Model, Conv2d, Linear, ReLu, Dropout2d, MaxPool2d, Sequential
from lightGE.utils import SGD, Trainer, nll_loss
import numpy as np
import gzip
from lightGE.data import DataLoader
from lightGE.core.mnist import MNIST
from lightGE.data.mnist import MnistDataset


def evaluate(model, dataset):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    correct = 0
    bar = tqdm.tqdm(dataloader)
    i = 0
    for x, y in bar:
        y_pred = model(Tensor(x))
        y_pred = np.argmax(y_pred.data, axis=1)
        y = np.argmax(y.data, axis=1)
        correct += np.sum(y_pred == y)
        i += 128
        bar.set_description("acc: {}".format(correct / i))
    return correct / len(dataset)


if __name__ == '__main__':
    mnist_dataset = MnistDataset()
    mnist_dataset.load_data('./res/mnist/')

    m = MNIST()
    opt = SGD(parameters=m.parameters(), lr=0.01)

    cache_path = 'tmp/mnist.pkl'
    train_dataset, eval_dataset = mnist_dataset.scaled_split(0.7)
    train_dataloader = DataLoader(train_dataset, 128, shuffle=False)
    eval_dataloader = DataLoader(eval_dataset, 128, shuffle=False)
    trainer = Trainer(model=m, optimizer=opt, loss_fun=nll_loss,
                      config={'batch_size': 128,
                              'epochs': 20,
                              'shuffle': False,
                              'save_path': cache_path})

    trainer.train(train_dataloader, eval_dataloader)
    # trainer.load_model('./tmp/mnist_bst.pkl')
    # trainer.train(train_dataset, eval_dataset)
    evaluate(m, eval_dataset)
