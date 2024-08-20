import json
import numpy as np
import torch
import argparse


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

from vector_db import process_pdfs_from_dataframe
from chain_inference import get_pipeline, langchain_inference

from utils import submission

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    database = process_pdfs_from_dataframe(**config["dataset"])

    model = AutoModelForCausalLM.from_pretrained(config["model"]["load_from"], 
                                                 quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base"], clean_up_tokenization_spaces=True)

    pipeline = get_pipeline(model, tokenizer, config["chain"]["pipeline"])

    results = langchain_inference(config["dataset"]["root"], database, pipeline)
    submission(results, config["exp"])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLM Template')

    parser.add_argument('-c', '--config', default='./inference_config.json', type=str,
                      help='config file path')

    args = parser.parse_args()


    with open(args.config, 'r') as f:
        config = json.load(f)

    main(config)
