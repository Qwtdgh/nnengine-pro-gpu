from lightGE.data.mnist import MnistDataset

from lightGE.utils import OptimizerFactory, Trainer, SGD, nll_loss
from lightGE.core import MNIST


class Trainer_Config:

    def __init__(self, dataset, dataDir, model, epochs, optimizer, loss, cache_path, sche, split=0.7, batch=10,
                 transformer=False):
        if transformer:
            pass
            # todo
        else:
            self.dataset = self.getDataset(dataset)
            self.dataset.load_data(dataDir)
            self.train_dataset, self.eval_dataset = self.dataset.split(split)
        self.model = self.getModel(model)
        self.opt = self.getOptimizer(optimizer)
        self.loss = eval(loss)
        self.sche = sche
        self.trainer = self.getTrainer(epochs, batch, cache_path)

    def getDataset(self, dataset):
        return eval(dataset)()

    def getModel(self, model):
        return eval(model)()

    def getOptimizer(self, optimizer):
        return OptimizerFactory(parameters=self.model.parameters(), lr=0.01).generate(optimizer)

    def getTrainer(self, epochs, batch, cache_path):
        trainer = Trainer(model=self.model, optimizer=self.opt, loss_fun=self.loss, schedule=self.sche,
                          config={'batch_size': batch,
                                  'epochs': epochs,
                                  'shuffle': False,
                                  'save_path': cache_path})
        return trainer
