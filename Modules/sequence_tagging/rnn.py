"""A Recurrent Neural Network tagger."""

# pylint: disable=redefined-outer-name

from logging import Logger
import torch
import torch.nn
from numpy import int64, mean
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..log import create_logger
from .brown import get_brown_tagged_sentences
from .util import (
    Arr,
    flat_pad_truncate_nested_list,
    pickler,
    to_data_loader,
)

rnn_logger = create_logger("rnn", "info")


class RNNTagger(torch.nn.Module):
    """A Recurrent Neural Network tagger."""

    def __init__(
        self,
        loss_fn: torch.nn.modules.loss._Loss,
        n_words: int,
        embedding_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        output_dim: int,
        logger: Logger = rnn_logger,
    ):
        super().__init__()  # type: ignore
        self.__loss_fn = loss_fn
        self.__n_words = n_words
        self.__embedding_dim = embedding_dim
        self.__hidden_dim = hidden_dim
        self.__hidden_layers = hidden_layers
        self.__output_dim = output_dim
        self.__logger = logger

        self.__embedding = torch.nn.Embedding(
            num_embeddings=self.__n_words, embedding_dim=self.__embedding_dim
        )
        logger.debug("__embedding %s", self.__embedding.weight.shape)

        self.__rnn = torch.nn.RNN(
            input_size=self.__embedding_dim,
            hidden_size=self.__hidden_dim,
            num_layers=self.__hidden_layers,
            batch_first=True,
        )
        logger.debug("__rnn %s", self.__rnn.weight_ih_l0.shape)

        self.__output = torch.nn.Linear(
            in_features=self.__hidden_dim, out_features=self.__output_dim
        )
        logger.debug("__output %s", self.__output.weight.shape)

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass."""

        # (batch size, sequence length)
        self.__logger.debug("inputs %s", inputs.shape)

        # Initialize the hidden-layer state.
        rnn_state_0 = torch.zeros(
            size=(self.__hidden_layers, inputs.shape[0], self.__rnn.hidden_size)
        )

        # Compute the embeddings of the input features.
        embeddings: Tensor = self.__embedding(inputs)
        # (batch size, sequence length, embedding dim)
        self.__logger.debug("embeddings %s", embeddings.shape)

        # Annotate types before unpacking.
        rnn_out: Tensor
        rnn_state: Tensor

        # Compute the output of the last layer of the RNN and the last hidden
        # state for each input in the batch.
        rnn_out, rnn_state = self.__rnn(embeddings, rnn_state_0.detach())

        # (batch size, sequence length, hidden dim)
        self.__logger.debug("rnn_out %s", rnn_out.shape)
        # (number of layers, hidden dim)
        self.__logger.debug("rnn_state %s", rnn_state.shape)

        # Compute the outputs.
        outputs: Tensor = self.__output(rnn_out)
        # (batch size, sequence length, output dim)
        self.__logger.debug("outputs %s", outputs.shape)

        # Transpose the outputs to match the labels.
        outputs_t = torch.transpose(outputs, 1, 2)
        # (batch size, output dim, sequence length)
        self.__logger.debug("outputs_t %s", outputs_t.shape)

        return outputs_t

    def train_(
        self,
        optimizer: torch.optim.Optimizer,
        n_epochs: int,
        train_loader: DataLoader[tuple[Tensor, ...]],
        val_loader: DataLoader[tuple[Tensor, ...]],
    ) -> None:
        """Train the model."""

        for epoch in range(n_epochs):
            train_correct = 0
            train_items = 0
            train_losses: list[float] = []

            self.train()

            inputs: Tensor
            labels: Tensor

            for inputs, labels in tqdm(train_loader):
                optimizer.zero_grad()

                outputs = self.forward(inputs)

                batch_loss = self.__loss_fn(outputs, labels)
                batch_loss.backward()  # type: ignore

                optimizer.step()

                train_losses.append(batch_loss.item())

                predicted = outputs.argmax(1)

                train_correct += (predicted == labels).sum().item()
                train_items += labels.size(0) * labels.size(1)

            train_accuracy = 100 * train_correct / train_items

            self.__logger.info("epoch %s", epoch + 1)
            self.__logger.info("training loss %.3f", mean(train_losses))
            self.__logger.info("training accuracy %.1f%%", train_accuracy)

            val_loss, val_accuracy = self.test_(val_loader)

            self.__logger.info("validation loss %.3f", val_loss)
            self.__logger.info("validation accuracy %.1f%%", val_accuracy)

    def test_(
        self, loader: DataLoader[tuple[Tensor, ...]]
    ) -> tuple[float, float]:
        """Test the model."""

        self.eval()

        test_correct = 0
        test_items = 0
        test_losses: list[float] = []

        for inputs, labels in loader:
            outputs = self.forward(inputs)

            batch_loss = self.__loss_fn(outputs, labels)

            test_losses.append(batch_loss.item())

            predicted = torch.argmax(outputs, dim=1)

            test_correct += (predicted == labels).sum().item()
            test_items += labels.size(0) * labels.size(1)

        test_accuracy = 100 * test_correct / test_items

        return mean(test_losses), test_accuracy


@pickler("brown_tagged_sentences_padded.pickle")
def get_brown_tagged_sentences_padded() -> (
    tuple[Arr[int64], Arr[int64], Arr[int64], Arr[int64], int, int]
):
    """This is extracted to a function to save re-computing it."""

    length = 40
    brown_tagged_sentences = get_brown_tagged_sentences()
    value = int64(brown_tagged_sentences.n_words)

    words_train_padded = flat_pad_truncate_nested_list(
        brown_tagged_sentences.words_train_encoded,
        length,
        value,
    )
    rnn_logger.info("words_train_padded %s", words_train_padded.shape)

    words_test_padded = flat_pad_truncate_nested_list(
        brown_tagged_sentences.words_test_encoded,
        length,
        value,
    )
    rnn_logger.info("words_test_padded %s", words_test_padded.shape)

    tags_train_padded = flat_pad_truncate_nested_list(
        brown_tagged_sentences.tags_train_encoded,
        length,
        value,
    )
    rnn_logger.info("tags_train_padded %s", tags_train_padded.shape)

    tags_test_padded = flat_pad_truncate_nested_list(
        brown_tagged_sentences.tags_test_encoded,
        length,
        value,
    )
    rnn_logger.info("tags_test_padded %s", tags_test_padded.shape)

    return (
        words_train_padded,
        words_test_padded,
        tags_train_padded,
        tags_test_padded,
        brown_tagged_sentences.n_words,
        brown_tagged_sentences.n_tags,
    )


if __name__ == "__main__":
    BATCH_SIZE = 64
    EMBEDDING_DIM = 25
    HIDDEN_DIM = 32
    HIDDEN_LAYERS = 1
    LEARNING_RATE = 0.0005
    N_EPOCHS = 10

    (
        words_train_padded,
        words_test_padded,
        tags_train_padded,
        tags_test_padded,
        n_words,
        n_tags,
    ) = get_brown_tagged_sentences_padded()

    data_loader_train = to_data_loader(
        words_train_padded, tags_train_padded, BATCH_SIZE
    )
    data_loader_test = to_data_loader(
        words_test_padded, tags_test_padded, BATCH_SIZE
    )

    # Ignore the padding index when computing the loss.
    cross_entropy_loss = torch.nn.CrossEntropyLoss(ignore_index=n_words)

    rnn_tagger = RNNTagger(
        loss_fn=cross_entropy_loss,
        # Include the padding index in the input.
        n_words=n_words + 1,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        hidden_layers=HIDDEN_LAYERS,
        # Include the padding index in the output.
        output_dim=n_tags + 1,
    )

    adam_optimizer = torch.optim.Adam(rnn_tagger.parameters(), lr=LEARNING_RATE)

    rnn_tagger.train_(
        n_epochs=N_EPOCHS,
        train_loader=data_loader_train,
        val_loader=data_loader_test,
        optimizer=adam_optimizer,
    )

    rnn_tagger.test_(data_loader_test)
