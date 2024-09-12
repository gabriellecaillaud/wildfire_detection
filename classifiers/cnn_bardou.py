import torch

from classifiers.nn_utils import Flattening, LocalResponseNorm, CNNLayer

class CNNBardou(torch.nn.Module):
    def __init__(self, input_size: tuple[int, int]) -> tuple[int, int]:
        """
        CNNBardou (Convolutional Neural Networks model from Bardou paper
        (Lung sounds classification using convolutional neural networks)) class constructor.

        Parameters
        ----------
        input_size : tuple
            -- Input size (length, height)
        """
        super(CNNBardou,self).__init__()

        # define the first Conv layer
        self.conv_layer1 = CNNLayer(
            in_channels=1,
            out_channels=64,
            cnn_kernel_size=(7, 7),
            cnn_stride=(1, 1),
            mp_kernel_size=(3, 3),
            mp_stride=(2, 2),
        )

        # define the second Conv layer
        self.conv_layer2 = CNNLayer(
            in_channels=64,
            out_channels=128,
            cnn_kernel_size=(5, 5),
            cnn_stride=(1, 1),
            mp_kernel_size=(3, 3),
            mp_stride=(2, 2),
        )

        # define the third Conv layer
        self.conv_layer3 = CNNLayer(
            in_channels=128,
            out_channels=256,
            cnn_kernel_size=(3, 3),
            cnn_stride=(1, 1),
            mp_kernel_size=(3, 3),
            mp_stride=(2, 2),
        )

        # define the forth Conv layer
        self.conv_layer4 = CNNLayer(
            in_channels=256,
            out_channels=384,
            cnn_kernel_size=(3, 3),
            cnn_stride=(1, 1),
            mp_kernel_size=(3, 3),
            mp_stride=(2, 2),
        )

        # define the fifth Conv layer
        self.conv_layer5 = CNNLayer(
            in_channels=384,
            out_channels=256,
            cnn_kernel_size=(3, 3),
            cnn_stride=(1, 1),
            mp_kernel_size=(3, 3),
            mp_stride=(2, 2),
        )

        # find the output size of the Conv layers 
        input_size = self.conv_layer1.get_output_size(input_size=input_size)
        input_size = self.conv_layer2.get_output_size(input_size=input_size)
        input_size = self.conv_layer3.get_output_size(input_size=input_size)
        input_size = self.conv_layer4.get_output_size(input_size=input_size)
        input_size = self.conv_layer5.get_output_size(input_size=input_size)

        # define a Flatten layer
        self.flatten = Flattening()

        # define a linear layers
        self.linear_layers = torch.nn.Sequential(
            # 1st linear layer
            torch.nn.Linear(
                in_features=input_size[0] * input_size[1] * 256, out_features=1000
            ),
            torch.nn.ReLU(),
            # 2nd linear layer
            torch.nn.Linear(in_features=1000, out_features=2),
        )

        # define normalization layer
        self.norm1 = LocalResponseNorm()
        self.norm2 = LocalResponseNorm()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method for the CNN model from Bardou.
        Computes and returns prediction of the model for the data x.

        Parameters
        ----------
        x : torch.Tensor
            -- Input data tensor
        """
        x = self.conv_layer1(x)
        x = self.norm1(x)
        x = self.conv_layer2(x)
        x = self.norm2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.conv_layer5(x)
        x = self.flatten(x)
        x = self.linear_layers(x)
        return x

