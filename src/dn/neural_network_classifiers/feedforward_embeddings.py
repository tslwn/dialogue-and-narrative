"""A feedforward text classifier with pre-trained embeddings."""

import gensim.downloader
from gensim.models import KeyedVectors
import torch
import torch.nn
from .feedforward import FeedforwardTextClassifier
from .tokenizer import Tokenizer


class FeedforwardTextClassifierEmbeddings(FeedforwardTextClassifier):
    """A feedforward text classifier with pre-trained embeddings."""

    def __init__(
        self,
        loss_fn: torch.nn.modules.loss._Loss,  # type: ignore
        num_embeddings: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        sequence_length: int,
        tokenizer: Tokenizer,
    ):
        super().__init__(
            loss_fn,
            num_embeddings,
            embedding_dim,
            hidden_dim,
            output_dim,
            sequence_length,
        )

        self.__embeddings: KeyedVectors = gensim.downloader.load(
            "glove-twitter-25"
        )

        self.__embedding_dim = self.__embeddings.vectors.shape[1]

        self.__embeddings_reindex = torch.zeros(
            (num_embeddings, self.__embedding_dim)
        )

        for word in tokenizer.vocabulary:
            if word in self.__embeddings:
                self.__embeddings_reindex[
                    tokenizer.vocabulary[word]
                ] = torch.from_numpy(  # type: ignore
                    self.__embeddings[word]
                )

        self._embedding_layer = torch.nn.Embedding.from_pretrained(  # type: ignore
            self.__embeddings_reindex, freeze=False
        )

        self._hidden_layer = torch.nn.Linear(
            in_features=sequence_length * embedding_dim, out_features=hidden_dim
        )

        self._activation = torch.nn.ReLU()

        self._output_features = torch.nn.Linear(
            in_features=hidden_dim, out_features=output_dim
        )
