import os.path as osp
from datasets import load_dataset

def get_dataset(root, split_ratio, tokenizer_args, tokenizer=None):
    
    assert tokenizer is not None, "Tokenizer must be assigned."

    data_path = osp.join(root,'train.csv')
    dataset = load_dataset('csv', data_files=data_path)
    dataset = dataset['train'].train_test_split(test_size=split_ratio)

    padding = tokenizer_args["padding"]
    truncation = tokenizer_args["truncation"]

    def tokenize_dataset(dataset):
        re_dataset = tokenizer(dataset['Question'], padding=padding, truncation=truncation)
        label = tokenizer(dataset['Answer'], padding=padding, truncation=truncation)
        re_dataset['labels'] = label['input_ids']
        return re_dataset

    dataset = dataset.map(tokenize_dataset, batched=True)

    return dataset
