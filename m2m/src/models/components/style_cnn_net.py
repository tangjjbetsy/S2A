import torch
import torch.nn as nn


class StyleCNNNet(nn.Module):
    def __init__(
        self,
        n_classes,
        num_of_feature,
        max_len,
        kernel_sizes,
        dropout,
        dense_size,
        activation="relu",
    ):
        super().__init__()

        # Select activation function based on input parameter
        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "rrelu":
            self.activation = nn.RReLU()
        else:
            raise ValueError("Unsupported activation function")

        # Construct the convolutional network layers using loops and parameter lists
        filters = [num_of_feature, 64, 64, 128, 128, 128]
        self.conv_layers = nn.Sequential()

        for i in range(len(kernel_sizes)):
            self.conv_layers.add_module(
                f"conv{i+1}", nn.Conv1d(filters[i], filters[i + 1], kernel_sizes[i], padding=1)
            )
            self.conv_layers.add_module(f"activation{i+1}", self.activation)
            self.conv_layers.add_module(f"batch_norm{i+1}", nn.BatchNorm1d(filters[i + 1]))
            if i % 2 == 1:  # Add pooling and dropout after every second conv layer
                self.conv_layers.add_module(
                    f"pool{i+1}", nn.MaxPool1d(kernel_sizes[i], stride=kernel_sizes[i])
                )
                self.conv_layers.add_module(f"dropout{i+1}", nn.Dropout(dropout))

        # Use a dummy input to determine the size of the fully connected layer
        with torch.no_grad():
            dummy_input = torch.ones([1, num_of_feature, max_len])
            out_shape = self.conv_layers(dummy_input).view(1, -1).shape[1]

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(out_shape, dense_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(dense_size, n_classes),
            nn.Softmax(dim=1),
        )

        self.n_classes = n_classes

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def _class_name(self):
        return "StyleCNNNet"
