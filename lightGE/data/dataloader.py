import numpy as np
from deprecated import deprecated


class Dataset(object):
    def __init__(self, data=None, label=None):
        self.x = data
        self.y = label

    def load_data(self, data_path):
        self.x: np.ndarray = np.load(data_path + 'data.npy')
        self.y: np.ndarray = np.load(data_path + 'label.npy')

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        return x, y

    def split(self, ratio):
        split_idx = int(len(self) * ratio)
        train_dataset = Dataset(self.x[:split_idx], self.y[:split_idx])
        test_dataset = Dataset(self.x[split_idx:], self.y[split_idx:])
        return train_dataset, test_dataset

    def scaled_split(self, ratio, scaled_ration = 0.5):
        split_idx = int(len(self) * ratio)
        split_step = int(1 / scaled_ration)
        train_dataset = Dataset(self.x[:split_idx:split_step], self.y[:split_idx:split_step])
        test_dataset = Dataset(self.x[split_idx::split_step], self.y[split_idx::split_step])
        return train_dataset, test_dataset

    @deprecated(version='1.0.0', reason="You should use function: padding")
    def padding_old(self, pad_size):
        padding_x = np.zeros((pad_size, *self.x.shape[1:]))
        padding_y = np.zeros((pad_size, *self.y.shape[1:]))
        self.x = np.concatenate((self.x, padding_x), axis=0)
        self.y = np.concatenate((self.y, padding_y), axis=0)

    def padding(self, pad_size, dim=0):
        pad_width_x = tuple((0, pad_size) if idx == dim else (0, 0) for idx in range(self.x.ndim))
        pad_width_y = tuple((0, pad_size) if idx == dim else (0, 0) for idx in range(self.y.ndim))
        self.x = np.pad(self.x, pad_width_x, 'constant')
        self.y = np.pad(self.y, pad_width_y, 'constant')


class DataLoader(object):
    def __init__(self, dataset, batch_size, shuffle=False, padding=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        if padding:
            self.padding()
        self.index = 0
        self.length = len(dataset)
        self.batch_num = self.length // self.batch_size
        self.indexes = np.arange(self.length)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < self.batch_num:
            batch_x = []
            batch_y = []
            for i in range(self.batch_size):
                idx = self.indexes[self.index * self.batch_size + i]
                x, y = self.dataset[idx]
                batch_x.append(x)
                batch_y.append(y)

            self.index += 1
            return np.array(batch_x), np.array(batch_y)
        else:
            self.index = 0
            if self.shuffle:
                np.random.shuffle(self.indexes)
            raise StopIteration

    def __len__(self):
        return self.batch_num

    def padding(self):
        padding_num = self.batch_size - self.length % self.batch_size
        self.dataset.padding(padding_num)
