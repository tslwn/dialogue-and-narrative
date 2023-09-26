"""A feedforward text classifier."""

from typing import Any
import numpy
import torch
import torch.nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from .tokenizer import DocumentTokens


def map_data_loader(
    documents: list[DocumentTokens], batch_size: int
) -> tuple[TensorDataset, DataLoader[tuple[Tensor, ...]]]:
    """Map a list of documents and tokens to a PyTorch data loader."""

    input_tensor = torch.from_numpy(  # type: ignore
        numpy.array([document["tokens"] for document in documents])
    )

    label_tensor = torch.from_numpy(  # type: ignore
        numpy.array([document["label"] for document in documents])
    )

    dataset = TensorDataset(input_tensor, label_tensor)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataset, loader


class FeedforwardTextClassifier(torch.nn.Module):
    """A feedforward text classifier."""

    def __init__(
        self,
        loss_fn: torch.nn.modules.loss._Loss,  # type: ignore
        num_embeddings: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        sequence_length: int,
    ):
        super().__init__()  # type: ignore

        self._loss_fn = loss_fn

        self._embedding_dim = embedding_dim

        self._embedding_layer = torch.nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim
        )

        self._hidden_layer = torch.nn.Linear(
            in_features=sequence_length * embedding_dim, out_features=hidden_dim
        )

        self._activation = torch.nn.ReLU()

        self._output_features = torch.nn.Linear(
            in_features=hidden_dim, out_features=output_dim
        )

    def forward(
        self, input_features: numpy.ndarray[Any, numpy.dtype[numpy.float64]]
    ):
        """Forward pass."""

        # The dimensions of the input features are: (batch size, sequence length).
        # The dimensions of the embeddings are: (batch size, sequence length, embedding dim).
        embedding_layer = self._embedding_layer(input_features)

        # Flatten the sequence of embedding vectors for each document into a single vector.
        # The dimensions of the embeddings are: (batch size, sequence length * embedding dim).
        embedding_layer = embedding_layer.reshape(
            embedding_layer.shape[0],
            input_features.shape[1] * self._embedding_dim,
        )

        # The dimensions of the hidden layer are: (batch size, hidden dim).
        hidden_layer = self._hidden_layer(embedding_layer)

        # The dimensions of the output are: (batch size, output dim).
        output_features = self._output_features(hidden_layer)

        return output_features

    def train_(
        self,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        train_loader: DataLoader[tuple[Tensor, ...]],
        dev_loader: DataLoader[tuple[Tensor, ...]],
    ) -> None:
        """Train the model."""

        for epoch in range(num_epochs):
            train_correct = 0
            train_items = 0
            train_losses: list[float] = []

            self.train()

            for input_features, labels in train_loader:
                optimizer.zero_grad()

                output_features = self.forward(input_features)

                batch_loss = self._loss_fn(output_features, labels)

                batch_loss.backward()  # type: ignore

                optimizer.step()

                train_losses.append(batch_loss.item())

                predicted = torch.argmax(output_features, dim=1)

                train_correct += (predicted == labels).sum().item()
                train_items += len(labels)

            train_accuracy = 100 * train_correct / train_items

            print(f"Epoch = {epoch}")
            print(f"Training loss = {numpy.mean(train_losses):.3f}")
            print(f"Training accuracy = {train_accuracy:.1f} %")

            dev_loss, dev_accuracy = self.test_(dev_loader)

            print(f"Validation loss = {dev_loss:.3f}")
            print(f"Validation accuracy = {dev_accuracy:.1f} %")

    def test_(
        self, loader: DataLoader[tuple[Tensor, ...]]
    ) -> tuple[float, float]:
        """Test the model."""

        self.eval()

        dev_correct = 0
        dev_items = 0
        dev_losses: list[float] = []

        for input_features, labels in loader:
            output_features = self.forward(input_features)

            batch_loss = self._loss_fn(output_features, labels)

            dev_losses.append(batch_loss.item())

            predicted = torch.argmax(output_features, dim=1)

            dev_correct += (predicted == labels).sum().item()
            dev_items += len(labels)

        dev_accuracy = 100 * dev_correct / dev_items

        return numpy.mean(dev_losses), dev_accuracy
