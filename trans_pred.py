from transformer.trans import predict

while True:
    english = input()
    out = predict(english, "res/simple_translation/simple_translation_en.txt.sentenceloader", "tmp/WMT18_model.pkl")
    print(out)