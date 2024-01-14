import numpy as np
from gensim.models import word2vec
from lightGE.data.segmentation.segmentation import segment_bpe, segment_jieba, segment_whitespace

import pickle


class SentenceLoader(object):
    def __init__(self, word_vector_size=512):
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

        pickle.dump(self, open(source_paths[0] + ".sentenceloader", "wb"))

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
