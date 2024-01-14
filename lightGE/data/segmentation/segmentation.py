import codecs

import jieba

from lightGE.data.segmentation.subword_nmt.learn_bpe import learn_bpe
from lightGE.data.segmentation.subword_nmt.apply_bpe import BPE


def segment_bpe(file_path: str, num_symbols=10000, min_frequency=2) -> [str, int, int]:
    segmented_path = file_path + ".segmented"
    codes_path = file_path + ".bpe.codes"
    max_sentence_len = 0
    sentence_num = 0

    file = codecs.open(file_path, encoding="utf-8")
    codes = codecs.open(codes_path, "w", encoding='utf-8')

    learn_bpe(file, codes, num_symbols, min_frequency=min_frequency)

    file.close()
    codes.close()

    file = codecs.open(file_path, encoding="utf-8")
    codes = codecs.open(codes_path, encoding='utf-8')
    segmented_file = codecs.open(segmented_path, "w", encoding='utf-8')

    bpe = BPE(codes)

    for line in file:
        write_line = add_start_end_token(bpe.process_line(line))
        max_sentence_len = max(max_sentence_len, len(write_line.split()))
        sentence_num += 1
        segmented_file.write(write_line)

    file.close()
    codes.close()
    segmented_file.close()
    return segmented_path, max_sentence_len, sentence_num


def segment_whitespace(file_paths: list[str]) -> [list[str], int, list[int]]:

    segmented_paths = []
    sentence_nums = []
    max_sentence_len = 0

    for file_path in file_paths:
        segmented_path = file_path + ".segmented"
        segmented_paths.append(segmented_path)

        sentence_num = 0

        file = open(file_path, "r", encoding="utf-8")
        segmented_file = open(segmented_path, "w", encoding="utf-8")

        for line in file:
            sentence_num += 1
            line = add_start_end_token(line)
            max_sentence_len = max(max_sentence_len, len(line.split()))
            segmented_file.write(line)
        sentence_nums.append(sentence_num)
    return segmented_paths, max_sentence_len, sentence_nums


def segment_jieba(file_paths: list[str]) -> [list[str], int]:

    segmented_paths = []
    max_sentence_len = 0

    for file_path in file_paths:
        segmented_path = file_path + ".segmented"
        segmented_paths.append(segmented_path)

        file = open(file_path, "r", encoding="utf-8")
        segmented_file = open(segmented_path, "w", encoding="utf-8")
        for line in file:
            cut_line = jieba.lcut(line)
            write_str = add_start_end_token(' '.join(cut_line))
            max_sentence_len = max(max_sentence_len, len(write_str.split()))
            segmented_file.write(write_str)

        file.close()
        segmented_file.close()
    return segmented_paths, max_sentence_len - 1


def add_start_end_token(sentence: str) -> str:
    out = ""

    leading_whitespace = len(sentence) - len(sentence.lstrip("\n\r "))
    if leading_whitespace:
        out += sentence[:leading_whitespace]

    out += '[START] ' + sentence.lstrip('\n\r ').rstrip('\n\r ') + ' [END]'

    tailing_whitespace = len(sentence) - len(sentence.rstrip('\n\r '))
    if tailing_whitespace:
        out += sentence[-tailing_whitespace:]

    return out