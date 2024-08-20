import json
import numpy as np
import torch
import argparse
import os
import os.path as osp
import time


from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from base_dataset import get_dataset
from base_trainer import get_trainer, get_lora_config

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):

    
    # 저장 되는 구조
    # exp(root) / exp_name / mmdd+time
    exp = config["experiment"]["name"]
    os.makedirs(osp.join('./exp', exp), exist_ok=True)
    exp_num = len(os.listdir(osp.join('./exp', exp)))
    save_path = osp.join('./exp', exp, str(exp_num))

    os.makedirs(save_path, exist_ok=True)

    
    # json도 배껴서 그냥 저장
    with open(osp.join(save_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    model_args = config["trainer"]["model"]
    model_args["torch_dtype"] = getattr(torch, model_args["torch_dtype"])

    # 모델 로드 및 양자화 설정 적용
    model = AutoModelForCausalLM.from_pretrained(**model_args)
    
     # 토크나이저 로드 및 설정
    tokenizer = AutoTokenizer.from_pretrained(model_args["pretrained_model_name_or_path"])


    dataset = get_dataset(**config['dataset'], tokenizer=tokenizer)

    lora_config = None
    if config["trainer"]["efficiency"]["use_lora"] == True:
        lora_config = get_lora_config(**config["trainer"]["efficiency"]["lora"])

    train_args = config["trainer"]["train_args"]
    train_args["output_dir"] = save_path

    trainer = get_trainer(dataset,model,tokenizer,train_args,lora_config)

    trainer.train()

    # model.save_pretrained(osp.join(save_path,"model"), from_pt=True) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLM Template')

    parser.add_argument('-c', '--config', default='./train_config.json', type=str,
                      help='config file path')

    args = parser.parse_args()


    with open(args.config, 'r') as f:
        config = json.load(f)
    
    main(config)
