import gzip

import numpy as np
import os
from PIL import Image
from deprecated import deprecated
from gensim.models import word2vec
from lightGE.data.segmentation.segmentation import segment_jieba, segment_whitespace


class Dataset(object):
    def __init__(self, data=None, label=None):
        self.x = data
        self.y = label
        self.index_to_label = dict()
        self.label_to_index = dict()
        self.label_index = 0
        self.shuffle = False

    def load_data(self, data_dir: str):
        self.x: np.array = np.load(os.path.join(data_dir, 'x.npy'))
        self.y: np.array = np.load(os.path.join(data_dir, 'y.npy'))

    def load_image_dir(self, image_dir: str, shuffle: bool = False):
        x = []
        y = []
        for sub_dir_name in os.listdir(image_dir):
            sub_dir = os.path.join(image_dir, sub_dir_name)
            if os.path.isdir(sub_dir):
                self.__add_new_label(sub_dir_name)
                for file_name in os.listdir(sub_dir):
                    image = Image.open(os.path.join(sub_dir, file_name))
                    x.append(np.array(image))
                    y.append(self.label_to_index[sub_dir_name])

        self.x = np.array(x)
        self.x = self.x.transpose(0, 3, 1, 2)  # (B, C, H, W)
        self.x = self.x / 255.  # 归一化

        y = np.array(y)
        y_vec = np.zeros((len(y), len(self.label_to_index)), dtype=np.float64)
        for i, label in enumerate(y):
            y_vec[i, label] = 1.0
        self.y = y_vec
        self.shuffle = shuffle

        # 数据集顺序打乱
        data_index = np.arange(self.x.shape[0])
        np.random.shuffle(data_index)
        self.x = self.x[data_index, :, :, :]
        self.y = self.y[data_index]

    def __add_new_label(self, label_name: str):
        if label_name not in self.label_to_index:
            self.label_to_index[label_name] = self.label_index
            self.index_to_label[self.label_index] = label_name
            self.label_index += 1

    def get_label(self, index: int):
        if index not in self.index_to_label:
            return ''
        else:
            return self.index_to_label[index]

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

    def scaled_split(self, ratio, scaled_ration=0.5):
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
        pad_width_x: tuple = tuple((0, pad_size) if idx == dim else (0, 0) for idx in range(self.x.ndim))
        pad_width_y: tuple = tuple((0, pad_size) if idx == dim else (0, 0) for idx in range(self.y.ndim))
        self.x = np.pad(self.x, pad_width_x, 'constant')
        self.y = np.pad(self.y, pad_width_y, 'constant')


class MnistDataset(Dataset):
    def __init__(self):
        super(MnistDataset, self).__init__()

    def load_data(self, data_dir):
        def extract_data(filename, num_data, head_size, data_size):
            with gzip.open(filename) as bytestream:
                bytestream.read(head_size)
                buf = bytestream.read(data_size * num_data)
                data = np.frombuffer(buf, dtype=np.uint8).astype(np.float64)
            return data

        data = extract_data(data_dir + 'train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
        trX = data.reshape((60000, 28, 28, 1))

        data = extract_data(data_dir + 'train-labels-idx1-ubyte.gz', 60000, 8, 1)
        trY = data.reshape((60000))

        data = extract_data(data_dir + 't10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
        teX = data.reshape((10000, 28, 28, 1))

        data = extract_data(data_dir + 't10k-labels-idx1-ubyte.gz', 10000, 8, 1)
        teY = data.reshape((10000))

        trY = np.asarray(trY)
        teY = np.asarray(teY)

        x = np.concatenate((trX, teX), axis=0)
        y = np.concatenate((trY, teY), axis=0).astype(np.int32)

        data_index = np.arange(x.shape[0])
        np.random.shuffle(data_index)
        # data_index = data_index[:128]
        x = x[data_index, :, :, :]
        y = y[data_index]
        y_vec = np.zeros((len(y), 10), dtype=np.float64)
        for i, label in enumerate(y):
            y_vec[i, y[i]] = 1.0

        x /= 255.
        x = x.transpose(0, 3, 1, 2)

        self.x = x
        self.y = y_vec


class DataLoader(object):
    def __init__(self):
        self.batch_num = 100
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < self.batch_num:
            return 0
        else:
            self.index = 0
            raise StopIteration

    def __len__(self):
        return self.batch_num


class ImageLoader(DataLoader):
    def __init__(self, batch_size=10, dataset=None, shuffle=False, padding=False):
        super(ImageLoader, self).__init__()
        self.dataset: Dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        if padding:
            self.padding()
        self.index = 0
        self.length = len(dataset)
        self.batch_num = self.length // self.batch_size
        self.indexes = np.arange(self.length)
        if self.shuffle or self.dataset.shuffle:
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


class SentenceLoader(DataLoader):
    def __init__(self, word_vector_size=512):
        super(SentenceLoader, self).__init__()
        self.batch_num = None
        self.batch_size = None
        self.word_vector_size = word_vector_size
        self.src_word2vec: word2vec.Word2Vec = word2vec.Word2Vec(vector_size=self.word_vector_size, min_count=1)
        self.tgt_word2vec: word2vec.Word2Vec = word2vec.Word2Vec(vector_size=self.word_vector_size, min_count=1)
        self.src_segmented_paths: list[str] = []
        self.tgt_segmented_paths: list[str] = []
        self.sentence_nums: list[int] = []
        self.src_max_sentence_len: int = 0
        self.tgt_max_sentence_len: int = 0
        self.max_sentence_len: int = 0

        self.idx = 0

    def load_sentences(self, source_paths: list[str], target_paths: list[str]):
        self.src_segmented_paths, self.src_max_sentence_len, self.sentence_nums = segment_whitespace(source_paths)
        self.tgt_segmented_paths, self.tgt_max_sentence_len = segment_jieba(target_paths)
        self.max_sentence_len = max(self.src_max_sentence_len, self.tgt_max_sentence_len)

        src_sentences = FileSentence(self.src_segmented_paths)
        self.src_word2vec.build_vocab(src_sentences)
        self.src_word2vec.train(src_sentences, total_examples=self.src_word2vec.corpus_count,
                                epochs=self.src_word2vec.epochs)

        tgt_sentences = FileSentence(self.tgt_segmented_paths)
        self.tgt_word2vec.build_vocab(tgt_sentences)
        self.tgt_word2vec.train(tgt_sentences, total_examples=self.tgt_word2vec.corpus_count,
                                epochs=self.tgt_word2vec.epochs)

    def src_sentence_matrix(self, sentence: str) -> np.array:
        if len(self.src_word2vec.wv.key_to_index) == 0:
            print("src_word2vec not trained !")
            return ""
        if not sentence.startswith("[START]"):
            sentence = "[START] " + sentence
        if not sentence.endswith("[END]"):
            sentence = sentence + "[END]"
        matrix = []
        for word in sentence.split():
            matrix.append(self.src_word2vec.wv[word])
        if len(matrix) < self.max_sentence_len:
            for _ in range(self.max_sentence_len - len(matrix)):
                matrix.append([0] * self.word_vector_size)
        return np.array([matrix])

    def tgt_sentence_matrix(self, sentence: str) -> np.array:
        if len(self.tgt_word2vec.wv.key_to_index) == 0:
            print("tgt_word2vec not trained !")
            return ""
        if not sentence.startswith("[START]"):
            sentence = "[START] " + sentence
        if not sentence.endswith("[END]"):
            sentence = sentence + "[END]"
        matrix = []
        for word in sentence.split():
            matrix.append(self.tgt_word2vec.wv[word])
        return np.array(matrix)

    def __iter__(self) -> [np.array, np.array]:
        src_file = open(self.src_segmented_paths[self.idx], "r", encoding="utf-8")
        tgt_file = open(self.tgt_segmented_paths[self.idx], "r", encoding="utf-8")
        src_batch, tgt_batch, tgt_label_batch = [], [], []
        src_batch_len, tgt_batch_len = [], []
        count = 0
        for src_line, tgt_line in zip(src_file, tgt_file):
            src_matrix, tgt_matrix, tgt_label_vector = [], [], []

            for src_word in src_line.strip().split():
                src_matrix.append(self.src_word2vec.wv[src_word])
            src_batch_len.append(len(src_matrix))
            padding_len = self.max_sentence_len - len(src_matrix)
            for _ in range(padding_len):
                src_matrix.append([0] * self.word_vector_size)

            for tgt_word in tgt_line.strip().split():
                tgt_matrix.append(self.tgt_word2vec.wv[tgt_word])
                label = [0] * len(self.tgt_word2vec.wv.key_to_index)
                label[self.tgt_word2vec.wv.key_to_index[tgt_word]] = 1
                tgt_label_vector.append(label)

            tgt_matrix = tgt_matrix[:-1]
            tgt_label_vector = tgt_label_vector[1:]
            tgt_batch_len.append(len(tgt_matrix))
            padding_len = self.max_sentence_len - len(tgt_matrix)
            for _ in range(padding_len):
                tgt_matrix.append([0] * self.word_vector_size)
                tgt_label_vector.append([0] * len(self.tgt_word2vec.wv.key_to_index))

            src_batch.append(src_matrix)
            tgt_batch.append(tgt_matrix)
            tgt_label_batch.append(tgt_label_vector)
            count += 1
            if count == self.batch_size:
                yield tuple((tuple((np.array(src_batch), np.array(tgt_batch))),
                             tuple((np.array(src_batch_len), np.array(tgt_batch_len))))), np.array(tgt_label_batch)
                src_batch.clear()
                tgt_batch.clear()
                tgt_label_batch.clear()
                count = 0

        src_file.close()
        tgt_file.close()

    def __len__(self):
        return int(self.sentence_nums[self.idx] / self.batch_size)

    def config(self, is_train: bool = True, batch_size=10):
        self.idx = 0 if is_train else 1
        self.batch_size = batch_size
        self.batch_num = int(self.sentence_nums[self.idx] / self.batch_size)
        return self


class FileSentence:
    def __init__(self, file_paths: list[str]):
        self.file_paths = file_paths

    def __iter__(self):
        for file_path in self.file_paths:
            for line in open(file_path, "r", encoding="utf-8"):
                yield line.strip().split()
