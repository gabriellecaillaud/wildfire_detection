from classifiers.cnn_bardou import CNNBardou
from classifiers.crnn_zhang import ConvolutionalRNNZhang
from datasets import ESCDataset
from features import Spectrogram, MelSpectrogram, Cochleagram
from models import train, FEATURES, generate_models

BATCH_SIZE = 50

MODEL_KWARGS = {"input_size": (256, 2206)} #size of a spectrogram/mel-spectrogram


esc_dataset = ESCDataset(download=False)



bardou_models = generate_models({"spectrogram": Spectrogram}, ConvolutionalRNNZhang, "cnn_bardou", classifier_kwargs=MODEL_KWARGS)
zhang_models = generate_models(FEATURES, ConvolutionalRNNZhang, "crnn_zhang", classifier_kwargs=MODEL_KWARGS)

loaders = esc_dataset.train_test_split().into_loaders(batch_size=BATCH_SIZE)

for model in bardou_models:
    train(model=model, loaders=loaders)
