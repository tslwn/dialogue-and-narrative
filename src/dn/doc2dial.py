"""Load the Doc2Dial dataset."""


import json
from typing import Any, Generator
import spacy
from dn.bigram_language_model.ngram_model import NgramModel
from dn.bigram_language_model.preprocessing import ngrams, pad


def load_dataset(count: int | None = None) -> list[str]:
    """Returns the utterances of the Doc2Dial dataset."""
    dialogues = [
        *generate_examples(
            # The filepath of the training data from the repository:
            # https://github.com/doc2dial/sharedtask-dialdoc2021
            "../../../sharedtask-dialdoc2021/data/doc2dial/v1.0.1/doc2dial_dial_train.json",
        )
    ]
    return [
        turn["utterance"]
        for dialogue in (dialogues[:count] if count is not None else dialogues)
        for turn in dialogue[1]["turns"]
    ]


def generate_examples(
    file: str,
) -> Generator[tuple[str, dict[Any, Any]], None, None]:
    """
    Adapted from https://huggingface.co/datasets/doc2dial/blob/main/doc2dial.py#L226.
    """
    with open(file, encoding="utf-8") as f:
        data = json.load(f)
        for domain in data["dial_data"]:
            for doc_id in data["dial_data"][domain]:
                for dialogue in data["dial_data"][domain][doc_id]:
                    obj = {
                        "dial_id": dialogue["dial_id"],
                        "domain": domain,
                        "doc_id": doc_id,
                        "turns": dialogue["turns"],
                    }

                    yield dialogue["dial_id"], obj


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    dataset = load_dataset(100)
    docs = [list(token.text for token in nlp(doc)) for doc in dataset]

    ORDER = 2
    model = NgramModel(order=ORDER, k=1.0)
    model.fit(docs)

    doc = list(
        pad(
            list(
                token.text
                for token in nlp("Can I do my DMV transactions online?")
            )
        )
    )

    print(f"\n{ORDER}-gram statistics:\n")
    for ngram in ngrams(ORDER, doc):
        token = ngram[-1]
        context = ngram[:-1]
        print(
            f"{' '.join(list(ngram))}".ljust(ORDER * 10)
            + f"{model.counts[(*context, token)]}".ljust(10)
            + f"{model.score(token, context):.4f}".ljust(10)
            + f"{model.logscore(token, context):.4f}".ljust(10)
        )

    print(f"\nPerplexity: {model.perplexity(doc):.4f}")
    print(f"\n{' '.join(doc + model.generate(100, seed=doc))}")
