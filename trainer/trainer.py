import pickle
import sys

import psutil
from torch import Tensor
import torch
from tqdm import tqdm
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import time
from datetime import datetime


from memory_profiler import profile
import gc

import logging

logging.basicConfig(level=logging.INFO)


class Trainer(object):

    def __init__(self, model=None, optimizer=None, loss_fun=None, config=None, schedule=None, load_path=None,
                 transformer=False):
        if load_path is not None:
            self.load_model(load_path)
        else:
            self.m = model
            self.opt = optimizer
            self.sche = schedule
            self.lf = loss_fun
            self.arr_train_loss = []
            self.arr_eval_loss = []
            self.epochs = config['epochs']
            self.batch_size = config['batch_size']
            self.shuffle = config['shuffle']
            self.save_path = config['save_path']
            self.transformer = transformer
            self.arr_lr = []

    def train(self, train_dataloader, eval_dataloader):
        min_eval_loss, best_epoch = float('inf'), 0
        for epoch_idx in range(self.epochs):
            st.write(f"Epoch: {epoch_idx}")
            train_loss = self._train_epoch(train_dataloader, epoch_idx)
            eval_loss = self._eval_epoch(eval_dataloader, epoch_idx)
            self.arr_train_loss.append(train_loss)
            self.arr_eval_loss.append(eval_loss)
            self.arr_lr.append(self.opt.param_groups[0]['lr'])
            if self.sche is not None:
                self.sche.step(eval_loss)

            logging.info("Lr: {}".format(self.opt.param_groups[0]['lr']))

            if eval_loss < min_eval_loss:
                min_eval_loss = eval_loss
                self.save_model(self.save_path)
                best_epoch = epoch_idx

        logging.info("Best epoch: {}, Best validation loss: {}".format(best_epoch, min_eval_loss))

        return min_eval_loss

    def _train_epoch(self, train_dataloader, epoch_idx) -> [float]:
        self.m.train()
        losses = []

        bar = tqdm(train_dataloader)
        batch_idx = 0
        text_output = st.empty()
        process_bar = st.progress(0)
        for batch_x, batch_y in bar:
            batch_idx += 1
            y_truth = torch.tensor(batch_y, requires_grad=False).to("cuda:3")
            if self.transformer is True:
                y_pred = self.m(self._get_batch_x_tensor(batch_x[0]), lens=batch_x[1])
            else:
                y_pred = self.m(self._get_batch_x_tensor(batch_x))
            loss: Tensor = self.lf(y_pred, y_truth)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.append(loss.data.cpu().numpy())

            description = "Epoch: {} ".format(epoch_idx) + \
                          "Batch: {}/{} ".format(batch_idx, len(train_dataloader)) + \
                          "Training loss: {} ".format(np.mean(losses)) + \
                          "lr: {} ".format(self.opt.param_groups[0]['lr'])

            bar.set_description(description)

            gc.collect()

            text_output.text(description)
            process_bar.progress(int(100 * batch_idx / train_dataloader.batch_num))
        logging.info("Epoch:{}, Training loss: {}".format(epoch_idx, np.mean(losses)))
        return np.mean(losses)

    def _eval_epoch(self, eval_dataloader, epoch_idx):
        self.m.eval()
        losses = []
        bar = tqdm(eval_dataloader)
        batch_idx = 0
        text_output = st.empty()
        process_bar = st.progress(0)
        for batch_x, batch_y in bar:
            batch_idx += 1
            y_truth = torch.tensor(batch_y, requires_grad=False).to("cuda:3")
            if self.transformer is True:
                y_pred = self.m(self._get_batch_x_tensor(batch_x[0]), lens=batch_x[1])
            else:
                y_pred = self.m(self._get_batch_x_tensor(batch_x))
            loss: Tensor = self.lf(y_pred, y_truth)
            losses.append(loss.data.cpu().numpy())

            description = "Epoch: {} ".format(epoch_idx) + \
                          "Evaluation: Batch: {}/{} ".format(batch_idx, len(eval_dataloader)) + \
                          'loss: {} '.format(np.mean(losses))

            bar.set_description(description)
            text_output.text(description)
            process_bar.progress(int(100 * batch_idx / eval_dataloader.batch_num))
            gc.collect()
        logging.info("Epoch:{}, Validation loss: {}".format(epoch_idx, np.mean(losses)))

        return np.mean(losses)

    def _get_batch_x_tensor(self, batch_x):
        if type(batch_x) is tuple:
            batch_x = list(batch_x)
            batch_x_tensor = []
            for x in batch_x:
                batch_x_tensor.append(torch.tensor(x, requires_grad=False).to("cuda:3"))
            return tuple(batch_x_tensor)
        else:
            return torch.tensor(batch_x, requires_grad=False).to("cuda:3")

    def load_model(self, cache_name):
        [self.m, self.opt, self.sche] = pickle.load(open(cache_name, 'rb'))

    def save_model(self, cache_name):
        pickle.dump([self.m, self.opt, self.sche], open(cache_name, 'wb'))

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
