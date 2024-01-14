import numpy as np
from tqdm import tqdm

from lightGE.core import Tensor
from lightGE.data import DataLoader
from lightGE.utils.trainer_config import Trainer_Config


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


class Trainer_Component:

    def __init__(self, trainer_config: Trainer_Config):
        self.trainer_config = trainer_config
        self.train_dataset = trainer_config.train_dataset
        self.eval_dataset = trainer_config.eval_dataset
        self.model = trainer_config.model
        self.trainer = trainer_config.trainer

    def train(self):
        self.trainer.train(DataLoader(self.train_dataset, self.trainer.batch_size),
                           DataLoader(self.eval_dataset, self.trainer.batch_size))
        evaluate(self.model, self.eval_dataset)

