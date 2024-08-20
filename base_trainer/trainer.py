from transformers import DataCollatorForLanguageModeling
from trl import SFTConfig, SFTTrainer


def get_trainer(dataset, model, tokenizer, config_args, lora_config = None):
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Masked Language Model이 아닌 Causal Language Model의 경우 False로 설정
    )

    # training_args = SFTConfig(
    #     output_dir='./results',              # 결과를 저장할 디렉토리
    #     eval_strategy="steps",         # 평가 전략을 "steps"로 설정
    #     eval_steps=50,                      # 평가 간격 (스텝 단위)
    #     learning_rate=2e-5,                  # 학습률
    #     per_device_train_batch_size=4,       # 훈련 배치 크기
    #     per_device_eval_batch_size=4,        # 평가 배치 크기
    #     num_train_epochs=5,                  # 에폭 수
    #     weight_decay=0.01,                   # 가중치 감소
    #     logging_dir='./logs',                # 로그 디렉토리
    #     logging_steps=10,                    # 로그 기록 간격
    #     prediction_loss_only=True,            # 생성 모델에서는 일반적으로 loss만을 예측
    #     max_seq_length=512
    # )
    
    training_args = SFTConfig(**config_args)

    trainer = SFTTrainer(
        model=model,                         # 모델
        args=training_args,                  # 훈련 인수
        train_dataset=dataset['train'],   # 훈련 데이터셋
        eval_dataset=dataset['test'] ,  # 평가 데이터셋
        tokenizer=tokenizer,                 # 토크나이저
        data_collator=data_collator,           # 데이터 콜레이터
        peft_config=lora_config
    )

    return trainer