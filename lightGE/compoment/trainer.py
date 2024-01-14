import pickle

from tqdm import tqdm
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import time
from datetime import datetime
import gc
from lightGE.core.tensor import Tensor, TcGraph
import logging
from lightGE.core.nn import Model
from lightGE.utils.optimizer import Optimizer
from lightGE.utils.scheduler import Scheduler
from lightGE.compoment.dataloader import DataLoader

logging.basicConfig(level=logging.INFO)


class Trainer(object):

    def __init__(self, model: Model = None, optimizer: Optimizer = None, scheduler: Scheduler = None, loss_fun=None,
                 epochs: int = 100, batch_size: int = 100, save_path: str = "./default.model",
                 train_dataloader: DataLoader = None, evaluate_dataloader: DataLoader = None,
                 is_transformer: bool = False):
        self.model = model
        self.best_model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_func = loss_fun
        self.epochs = epochs
        self.batch_size = batch_size
        self.save_path = save_path
        self.train_dataloader = train_dataloader
        self.evaluate_dataloader = evaluate_dataloader
        self.is_transformer = is_transformer

        self.arr_train_loss = []
        self.arr_eval_loss = []

        self.optimizer.set_parameters(self.model.parameters())

    def train(self):
        print(type(self.model))

        min_eval_loss, best_epoch = float('inf'), 0
        for epoch_idx in range(self.epochs):
            st.write(f"Epoch: {epoch_idx}")
            train_loss = self._train_epoch(self.train_dataloader, epoch_idx)
            eval_loss = self._eval_epoch(self.evaluate_dataloader, epoch_idx)
            self.arr_train_loss.append(train_loss)
            self.arr_eval_loss.append(eval_loss)
            if self.scheduler is not None:
                self.scheduler.step(eval_loss)

            logging.info("Lr: {}".format(self.optimizer.lr))

            if eval_loss < min_eval_loss:
                min_eval_loss = eval_loss
                self.best_model = self.model
                best_epoch = epoch_idx

        logging.info("Best epoch: {}, Best validation loss: {}".format(best_epoch, min_eval_loss))
        st.write("Best epoch: {}, Best validation loss: {}".format(best_epoch, min_eval_loss))

        return min_eval_loss

    def _train_epoch(self, train_dataloader: DataLoader, epoch_idx) -> [float]:
        self.model.train()
        losses = []

        bar = tqdm(train_dataloader)
        batch_idx = 0
        text_output = st.empty()
        process_bar = st.progress(0)
        for batch_x, batch_y in bar:
            batch_idx += 1
            y_truth = Tensor(batch_y, autograd=False)
            if self.is_transformer:
                y_prediction = self.model(get_batch_x_tensor(batch_x[0]), lens=batch_x[1])
            else:
                y_prediction = self.model(get_batch_x_tensor(batch_x))
            loss: Tensor = self.loss_func(y_prediction, y_truth)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.data)

            description = "Epoch: {} ".format(epoch_idx) + \
                          "Batch: {}/{} ".format(batch_idx, len(train_dataloader)) + \
                          "Training loss: {} ".format(np.mean(losses)) + \
                          "lr: {} ".format(self.optimizer.lr) + \
                          "step: {} ".format(self.optimizer.t)

            bar.set_description(description)

            TcGraph.Clear()
            gc.collect()

            text_output.text(description)
            process_bar.progress(int(100 * batch_idx / train_dataloader.batch_num))
        logging.info("Epoch:{}, Training loss: {}".format(epoch_idx, np.mean(losses)))
        return np.mean(losses)

    def _eval_epoch(self, eval_dataloader: DataLoader, epoch_idx):
        self.model.eval()
        losses = []
        bar = tqdm(eval_dataloader)
        batch_idx = 0
        text_output = st.empty()
        process_bar = st.progress(0)
        for batch_x, batch_y in bar:
            batch_idx += 1
            y_truth = Tensor(batch_y, autograd=False)
            if self.is_transformer:
                y_prediction = self.model(get_batch_x_tensor(batch_x[0]), lens=batch_x[1])
            else:
                y_prediction = self.model(get_batch_x_tensor(batch_x))
            loss: Tensor = self.loss_func(y_prediction, y_truth)
            losses.append(loss.data)

            description = "Epoch: {} ".format(epoch_idx) + \
                          "Evaluation: Batch: {}/{} ".format(batch_idx, len(eval_dataloader)) + \
                          'loss: {} '.format(np.mean(losses))

            bar.set_description(description)
            text_output.text(description)
            process_bar.progress(int(100 * batch_idx / eval_dataloader.batch_num))
            TcGraph.Clear()
            gc.collect()
        logging.info("Epoch:{}, Validation loss: {}".format(epoch_idx, np.mean(losses)))

        return np.mean(losses)

    def save_img(self):
        x = range(self.epochs)
        plt.title("Train Result")
        plt.plot(x, self.arr_train_loss, color='red', label='train')
        plt.plot(x, self.arr_eval_loss, color='blue', label='validation')
        plt.xticks(range(0, self.epochs, self.epochs // 10))
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        name = "tmp/epoch-{}_{}.jpg".format(self.epochs, datetime.fromtimestamp(time.time())).replace(":", "-")
        plt.savefig(name)
        plt.show()

def get_batch_x_tensor(batch_x):
    if type(batch_x) is tuple:
        batch_x = list(batch_x)
        batch_x_tensor = []
        for x in batch_x:
            batch_x_tensor.append(Tensor(x, autograd=False))
        return tuple(batch_x_tensor)
    else:
        return Tensor(batch_x, autograd=False)