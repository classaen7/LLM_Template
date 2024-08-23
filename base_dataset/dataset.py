import os.path as osp
from datasets import load_dataset

def get_dataset(root, split_ratio, tokenizer_args, tokenizer=None):
    
    assert tokenizer is not None, "Tokenizer must be assigned."

    data_path = osp.join(root,'train.csv')
    dataset = load_dataset('csv', data_files=data_path)
    dataset = dataset['train'].train_test_split(test_size=split_ratio)

    padding = tokenizer_args["padding"]
    truncation = tokenizer_args["truncation"]

    # def tokenize_dataset(dataset):
    #     re_dataset = tokenizer(dataset['Question'], padding=padding, truncation=truncation)
    #     label = tokenizer(dataset['Answer'], padding=padding, truncation=truncation)
    #     re_dataset['labels'] = label['input_ids']
    #     return re_dataset
    
    def tokenize_dataset(dataset):
        # 각 질문에 대해 [INST]와 [/INST] 토큰 추가
        questions_with_tokens = ["[INST] " + q + " [/INST]" for q in dataset['Question']]
        
        # Tokenizer를 사용해 Question과 Answer를 각각 토큰화
        re_dataset = tokenizer(questions_with_tokens, padding="max_length", truncation=True)
        label = tokenizer(dataset['Answer'], padding="max_length", truncation=True)
        
        # Label로 Answer의 input_ids를 추가
        re_dataset['labels'] = label['input_ids']
        
        return re_dataset


    dataset = dataset.map(tokenize_dataset, batched=True)

    return dataset

