import copy
import pickle

from lightGE.data.sentenceloader import SentenceLoader
from lightGE.core.transformer.transformer import Transformer
from lightGE.utils.optimizer import Adam, SGD
from lightGE.utils.trainer import Trainer
from lightGE.utils.loss import multi_classification_cross_entropy_loss

if __name__ == "__main__":
    word_vector_size = 512
    n_heads = 8
    epochs = 1000
    batch = 100

    sentence_loader = SentenceLoader(word_vector_size=word_vector_size)
    sentence_loader.load_sentences(
        ['res/simple_translation/simple_translation_en.txt', 'res/simple_translation/val_simple_translation_en.txt'],
        ['res/simple_translation/simple_translation_zh.txt', 'res/simple_translation/val_simple_translation_zh.txt'])
    # sentence_loader.load_sentences(["res/WMT18/WMT18-news-commentary-en.txt", "res/WMT18/WMT18-news-commentary-en.txt"],
    #                                ["res/WMT18/WMT18-news-commentary-zh.txt", "res/WMT18/WMT18-news-commentary-zh.txt"])

    train_sentence_loader = copy.deepcopy(sentence_loader).config(True, batch_size=batch)
    eval_sentence_loader = copy.deepcopy(sentence_loader).config(False, batch_size=batch)
    model = Transformer(batch=batch, sentence_len=train_sentence_loader.max_sentence_len,
                        n_inputs=word_vector_size,
                        n_heads=n_heads, vocab_len=len(train_sentence_loader.tgt_word2vec.wv.key_to_index),
                        hidden_feedforward=2048)
    optimizer = Adam(parameters=model.parameters(), beta=(0.9, 0.98), d_model=word_vector_size,
                     warmup_step=400)
    model_save_path = "tmp/WMT18_model.pkl"
    trainer = Trainer(model=model, optimizer=optimizer, loss_fun=multi_classification_cross_entropy_loss,
                      transformer=True,
                      config={
                          'batch_size': batch,
                          'epochs': epochs,
                          'shuffle': False,
                          'save_path': model_save_path
                      })
    trainer.train(train_sentence_loader, eval_sentence_loader)
