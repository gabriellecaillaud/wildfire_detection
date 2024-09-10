import logging
import json

import torch
from urllib.request import urlopen
from urllib.parse import urlencode

from classifiers.nn_utils import SequentialSaveableModel, SaveableModel
from datasets import TrainValidTestDataLoader
from features import Spectrogram, MelSpectrogram, Cochleagram



FEATURES = {
    "spectrogram": Spectrogram,
    "mel_spectrogram": MelSpectrogram,
# "cochleagram": Cochleagram
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def notify(*msg: str):
    message = " ".join(msg)
    print(message)

notify("Job started!")

def train(
    model: SaveableModel,
    loaders: TrainValidTestDataLoader,
    epochs: int = 100,
    learning_rate: float = 0.001,
    logging_file: str = "classifiers.log",
):
    """Train a neural network on the ESC-50 dataset.

    Parameters
    ----------
    model: torch.nn.Module
        the model to train on the ESC-50 dataset
    train_percentage: float
        the percentage data to use for training
    test_percentage: float
        the percentage data to use for testing
    learning_rate: float
        the learning rate to use for training
    """

    logging.basicConfig(filename=logging_file)

    # We use cross-entropy as it is well-known for performing well in classification problems
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    

    for epoch in range(epochs):
        notify(f"----------------------- EPOCH {epoch} -----------------------")

        running_loss = 0.0

        for batch_num, data in enumerate(loaders.train):
            waveforms, labels = data
            waveforms, labels = waveforms.to(device), labels.to(device)

            # set optimizer params to zero
            optimizer.zero_grad()

            predictions = model(waveforms)

            loss = loss_func(predictions, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()


        notify(f"Batch {batch_num}, loss: {running_loss}")

        confusion = torch.zeros((50,50), dtype=torch.int16)

        with torch.set_grad_enabled(False):
            for batch_num, data in enumerate(loaders.valid):
                waveforms, labels = data
                waveforms, labels = waveforms.to(device), labels.to(device)
                
                outputs = model(waveforms).tolist()
                labels = labels.tolist()

                for pred, truth in zip(outputs, labels):
                    confusion[pred, truth] += 1

        true_preds = torch.sum(torch.diagonal(confusion))

        accuracy = true_preds/torch.sum(confusion)

        notify(f"Accuracy: {accuracy*100:.2f}%")

        torch.set_printoptions(profile="full")
        print(confusion)
        torch.set_printoptions(profile="default")
                



        model.save(epoch=epoch)

    notify("----------------------FINISHED TRAINING----------------------")


def generate_models(
    feature_extractions: dict[str, torch.nn.Module],
    classifier: torch.nn.Module,
    classifier_name: str,
    classifier_kwargs: dict
) -> SequentialSaveableModel:
    """Generates models for all given feature extractions given a classifier

    Parameters
    ----------
    feature_extractions: list[str, torch.nn.Module]
        a list of feature extractions with their names
    classifier: torch.nn.Module
        the classifier to use
    classifier_name: the name of the classifier

    Returns
    list[SaveableModel]
        a list of saveable models, one for each feature extraction
    """

    models = []
    for name, feature_extraction in feature_extractions.items():
        
        model = SequentialSaveableModel(
            (feature_extraction(), name), (classifier(**classifier_kwargs), classifier_name)
        )

        model.to(device)
        models.append(model)

    return models
