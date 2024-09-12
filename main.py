from classifiers.cnn_bardou import CNNBardou
from classifiers.crnn_zhang import ConvolutionalRNNZhang
from classifiers.mlp_post_ast import MLPPostAst
from datasets import ESCDataset, ESCDatasetBin, ESCDatasetBinNoOverlap
from features import Spectrogram, MelSpectrogram, Cochleagram, AstEncoder
from models import train, FEATURES, generate_models

BATCH_SIZE = 32

MODEL_KWARGS = {"input_size": (256, 2206)} #size of a spectrogram/mel-spectrogram

APPROACH = 2  # change the approach based on 2, 10 or 50 classes

esc_dataset = ESCDataset(download=False, data_size=100) if APPROACH!=2 else ESCDatasetBin(download=False,data_size=100, train_index= list(range(2744)))

print(f"We are using {esc_dataset} for the dataset")
print(f"Type of approach : {APPROACH}")

models = generate_models({"spectrogram": Spectrogram}, CNNBardou, "cnn_bardou")
# models = generate_models(FEATURES, ConvolutionalRNNZhang, "crnn_zhang", classifier_kwargs=MODEL_KWARGS)
# models = generate_models({"ast_Encoder": AstEncoder}, MLPPostAst, "MLP_post_AST")

loaders = esc_dataset.train_test_split().into_loaders(batch_size=BATCH_SIZE)

for model in models:
    train(model=model, loaders=loaders)
