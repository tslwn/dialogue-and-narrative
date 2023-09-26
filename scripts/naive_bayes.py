"""Naive Bayes classifier."""

from math import log

Document = tuple[list[str], str]


def get_vocab(documents: list[Document]) -> set[str]:
    """Get the vocabulary of the documents."""
    return set(
        token
        for document, _ in documents
        for tokens in document
        for token in tokens
    )


def get_documents_in_class(
    documents: list[Document], class_name: str
) -> list[list[str]]:
    """Get the documents in the class."""
    return [
        document
        for document, document_class_name in documents
        if document_class_name == class_name
    ]


def train_naive_bayes(
    documents: list[Document], class_names: list[str], k: float
) -> tuple[dict[str, float], dict[str, dict[str, float]], set[str]]:
    """Train the Naive Bayes classifier."""

    # The number of documents.
    n_documents = len(documents)

    # The vocabulary of the documents.
    vocab = get_vocab(documents)

    # The documents in each class.
    documents_by_class: dict[str, list[list[str]]] = {}

    # The token counts in each class.
    token_counts_by_class: dict[str, dict[str, int]] = {}

    # The log prior probability of each class.
    log_prior_by_class: dict[str, float] = {}

    # The log likelihood of each token in each class.
    log_likelihood_by_class: dict[str, dict[str, float]] = {}

    # For each class...
    for class_name in class_names:
        # The documents in the class.
        documents_by_class[class_name] = get_documents_in_class(
            documents, class_name
        )

        # The number of documents in the class.
        n_c = len(documents_by_class[class_name])

        # The log prior probability of the class.
        log_prior_by_class[class_name] = log(n_c / n_documents)

        # For each document in the class...
        for document in documents_by_class[class_name]:
            # For each token in the document...
            for token in document:
                # Increment the count in the class.
                token_counts_by_class[class_name][token] += 1

        for token in vocab:
            log_likelihood_by_class[class_name][token] = log(
                (token_counts_by_class[class_name][token] + k)
                / sum(
                    token_counts_by_class[class_name][token] + k
                    for token in vocab
                )
            )

    return log_prior_by_class, log_likelihood_by_class, vocab
