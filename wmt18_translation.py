import copy

from torch.optim import Adam

from dataloader.sentenceloader import SentenceLoader
from loss.loss import multi_classification_cross_entropy_loss
from trainer.trainer import Trainer
from transformer.transformer import Transformer


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
    model.to("cuda:3")
    # for name, param in model.named_parameters():
    #     print(name, param.device)
    # exit(0)
    optimizer = Adam(params=model.parameters(), lr=0.0001, betas=(0.9, 0.98))
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
