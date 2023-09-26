"""A script to complete the fifth notebook exercise."""

import numpy
import torch
from ..datasets import TweetEvalDataset
from .feedforward import FeedforwardTextClassifier, map_data_loader
from .feedforward_embeddings import FeedforwardTextClassifierEmbeddings
from .tokenizer import DocumentTokens, Tokenizer


def get_data_loader(
    split: str,
    tokenizer: Tokenizer | None = None,
    sequence_length: int = 40,
    batch_size: int = 64,
):
    """Prepare a data loader."""

    data = list(TweetEvalDataset("emotion", split).iter())

    if tokenizer is None:
        tokenizer = Tokenizer(data)

    tokens = list(tokenizer.map(data))

    padded = tokenizer.pad(tokens, sequence_length)

    _, data_loader = map_data_loader(padded, batch_size)

    return data, data_loader, tokenizer


def prepare_data(sequence_length: int = 40, batch_size: int = 64):
    """Prepare data loaders for training, testing, and validation."""

    train, train_loader, tokenizer = get_data_loader(
        "train",
        tokenizer=None,
        sequence_length=sequence_length,
        batch_size=batch_size,
    )

    _, test_loader, _ = get_data_loader(
        "test",
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        batch_size=batch_size,
    )

    _, dev_loader, _ = get_data_loader(
        "validation",
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        batch_size=batch_size,
    )

    num_embeddings = len(tokenizer.vocabulary) + 1

    train_labels = [document["label"] for document in train]
    output_dim = numpy.unique(train_labels).shape[0]

    return (
        tokenizer,
        train_loader,
        test_loader,
        dev_loader,
        num_embeddings,
        output_dim,
    )


def print_length_statistics(documents: list[DocumentTokens]) -> None:
    """Print statistics about the lengths of the documents."""

    lengths = [len(document["tokens"]) for document in documents]

    print(f"\tMean = {numpy.mean(lengths):.3f}")
    print(f"\tMedian = {numpy.median(lengths):.3f}")
    print(f"\tMax = {numpy.max(lengths)}")  # type: ignore


def train_and_test(
    embeddings: bool = False,
    sequence_length: int = 40,
    batch_size: int = 64,
    embedding_dim: int = 25,
    hidden_dim: int = 32,
    learning_rate: float = 0.0005,
    num_epochs: int = 10,
) -> None:
    """Train and test feedforward neural-network classifier."""

    (
        tokenizer,
        train_loader,
        test_loader,
        dev_loader,
        num_embeddings,
        output_dim,
    ) = prepare_data(sequence_length=sequence_length, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()

    if embeddings:
        model = FeedforwardTextClassifierEmbeddings(
            loss_fn=loss_fn,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            sequence_length=sequence_length,
            tokenizer=tokenizer,
        )
    else:
        model = FeedforwardTextClassifier(
            loss_fn=loss_fn,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            sequence_length=sequence_length,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train_(
        num_epochs=num_epochs,
        train_loader=train_loader,
        dev_loader=dev_loader,
        optimizer=optimizer,
    )

    model.test_(loader=test_loader)


if __name__ == "__main__":
    train_and_test(embeddings=False)
    train_and_test(embeddings=True)
