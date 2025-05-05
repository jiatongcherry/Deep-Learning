from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from transformers import BertTokenizerFast

"""
    Align labels with tokenized word IDs, ensuring special tokens are ignored (-100).

    The first rule we’ll apply is that special tokens get a label of -100.
    This is because by default -100 is an index that is ignored in the loss function we will use (cross entropy).
    Then, each token gets the same label as the token that started the word it’s inside, since they are part of the same entity.
    For tokens inside a word but not at the beginning, we replace the B- with I- (since the token does not begin the entity):

    Args:
        labels: List of token labels.
        word_ids: List of word IDs.

    Returns:
        A list of aligned labels.
"""

def align_labels_with_tokens(
    labels: list[int],
    word_ids: list[int | None],
) -> list[int]:
    """
        Align labels with tokenized word IDs, ensuring special tokens are ignored (-100).
    """
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


def build_dataset() -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    """
    Build the dataset.

    Returns:
        The dataset.


    Below is an example of how to load a dataset.

    ```python
    from datasets import load_dataset

    raw_datasets = load_dataset('tomaarsen/MultiCoNER', 'multi')
    ```

    You can replace this with your own dataset. Make sure to include
    the `test` split and ensure that it is the same as the test split from the MultiCoNER NER dataset,
    Which means that:
        raw_datasets["test"] = load_dataset('tomaarsen/MultiCoNER', 'multi', split="test")
    """
    # Write your code here.
    raw_datasets = load_dataset('tomaarsen/MultiCoNER', 'multi')
    raw_datasets["test"] = load_dataset('tomaarsen/MultiCoNER', 'multi', split="test")
    return raw_datasets

def tokenize_and_align_labels(examples: dict, tokenizer) -> dict:
    """
    Tokenize input examples and align labels for token classification.

    To preprocess our whole dataset, we need to tokenize all the inputs and apply align_labels_with_tokens() on all the labels.
    To take advantage of the speed of our fast tokenizer, it’s best to tokenize lots of texts at the same time,
    so this function will processes a list of examples and return a list of tokenized inputs with aligned labels.

    Args:
        examples: Input examples.
        tokenizer: Tokenizer object.

    Returns:
        Tokenized inputs with aligned labels.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

def preprocess_data(raw_datasets: DatasetDict, tokenizer) -> DatasetDict:
    """
    Preprocess the data.

    Args:
        raw_datasets: Raw datasets.
        tokenizer: Tokenizer object.

    Returns:
        Tokenized datasets.
    """
    tokenized_datasets: DatasetDict = raw_datasets.map(
        function=lambda examples: tokenize_and_align_labels(
            examples=examples,
            tokenizer=tokenizer,
        ),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    return tokenized_datasets


tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
raw_datasets = build_dataset()
ner_feature = raw_datasets["train"].features["ner_tags"]
label_names = ner_feature.feature.names

# if __name__ == "__main__":
#     # print("Raw Datasets:")
#     # print(raw_datasets)
#     # print(raw_datasets["train"][0]["tokens"])
#     # print(raw_datasets["train"][0]["ner_tags"])
#
#     # words = raw_datasets["train"][0]["tokens"]
#     # labels = raw_datasets["train"][0]["ner_tags"]
#     # line1 = ""
#     # line2 = ""
#     # for word, label in zip(words, labels):
#     #     full_label = label_names[label]
#     #     max_length = max(len(word), len(full_label))
#     #     line1 += word + " " * (max_length - len(word) + 1)
#     #     line2 += full_label + " " * (max_length - len(full_label) + 1)
#     #
#     # print(line1)
#     # print(line2)
#
#     # Initialize the tokenizer
#     tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
#     raw_datasets = build_dataset()
#
#     # Extract NER feature and label names
#     ner_feature = raw_datasets["train"].features["ner_tags"]
#     label_names = ner_feature.feature.names
#
#     #Preprocess the data
#     tokenized_datasets = preprocess_data(raw_datasets, tokenizer)
#
#     print("\nSample Tokenized Input:")
#     sample = tokenized_datasets["train"][0]
#     print("Tokens:", sample['input_ids'])
#     print("Aligned Labels:", sample['labels'])