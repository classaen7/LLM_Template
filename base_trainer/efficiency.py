from peft import LoraConfig

# lora_config = LoraConfig(
#     r=32,
#     # target_modules=['o_proj', 'q_proj', 'up_proj', 'v_proj', 'k_proj', 'down_proj', 'gate_proj'],
#     task_type="CAUSAL_LM"
# )

def get_lora_config(**lora_args):
    return LoraConfig(**lora_args)

# bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16
#     )
