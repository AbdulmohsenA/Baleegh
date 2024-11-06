import modal
import torch
import wandb
import os
import numpy as np
from huggingface_hub import HfApi, HfFolder
import transformers
from tqdm import tqdm
from trl import DPOConfig, DPOTrainer, SFTTrainer, SFTConfig, KTOConfig, KTOTrainer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
from transformers import DataCollatorForSeq2Seq, BitsAndBytesConfig
from datasets import load_dataset
from evaluate import load
from transformers import get_scheduler
from datetime import datetime

app = modal.App("DPO")

image = modal.Image.debian_slim().pip_install([
    "transformers", "torch", "wandb", "evaluate", "huggingface_hub", "datasets", "numpy", "trl"
])


def setup_environment(wandb_run_id=None):
    HfFolder.save_token('REDACTED')
    wandb.login(key='REDACTED')
    seed = 1
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)
    np.random.seed(seed)
    
    if wandb_run_id is not None:
        wandb.init(project="Baleegh", id=wandb_run_id, resume="must")
        print(f"Resuming run: {wandb_run_id}")
    # print(f"GPU is available: {torch.cuda.is_available()}")
    # print(torch.cuda.get_device_name(0))

def load_model_and_tokenizer():
    try:
        artifact = wandb.use_artifact('DPO:latest', type='model')
        model_dir = artifact.download()
    except Exception as e:
        import warnings
        warnings.warn("No WandB logged model found. Using Base Model")
        model_dir = "AbdulmohsenA/Faseeh"

    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, src_lang="eng_Latn", tgt_lang="arb_Arab")

    return model, tokenizer

def sanity_check(model, tokenizer):

    texts = [
    "And the Egyptian foreign minister ordered the citizens to stick together.",
    "Hello! It's been a while since we last spoke.",
    "We should stay together hands on hands.",
    "Could you please help me with this task?",
    "Thank you so much for your kindness and support.",
    "Can you pass me the salt, please?",
    "I would rather stay home and read a good book tonight.",
    "I’m sorry for the misunderstanding. It wasn’t my intention.",
    "The sky is so clear and beautiful today.",
    "If I were you, I would reconsider that decision.",
    "He thought for a moment, then replied, 'I believe this is the best choice.'"
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to('cuda')

    model.eval()
    with torch.inference_mode():
        output = model.generate(**inputs)
        preds = tokenizer.batch_decode(output, skip_special_tokens=True)
        
    return [[input_sentence, output_sentence] for input_sentence, output_sentence in list(zip(texts, preds))]


# @app.function(gpu=modal.gpu.A100(size="80GB"), image=image, timeout=10000)
@app.function(gpu=modal.gpu.A10G(), image=image, timeout=10000)
def train_model(wandb_run_id):
    setup_environment(wandb_run_id)
    
    model, tokenizer = load_model_and_tokenizer()
    model_dir = "./logs/model_weights"
    artifact_name = "DPO"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)


    print("Sanity Check:")
    example = sanity_check(model, tokenizer)[0]
    print(f"Input: {example[0]}")
    print(f"Output: {example[1]}")

    # Train DPO
    dataset = load_dataset("Abdulmohsena/Pairs_DPO")['train'].shuffle()
    
    # https://thisisrishi.medium.com/direct-preference-optimization-dpo-in-llms-21225b991f4e
    

    # train_args = KTOConfig(
    #     output_dir=model_dir,
    #     logging_steps=10,
    #     per_device_train_batch_size=6,
    #     learning_rate=1e-6
    # )
    
    # trainer = KTOTrainer(
    #     model=model,
    #     args=train_args,
    #     processing_class=tokenizer,
    #     train_dataset=dataset
    # )
    # # Configure DPO training with memory optimizations
    training_args = DPOConfig(
        output_dir="FASEEHDPO",
        per_device_train_batch_size=6,  # Reduced batch size
        logging_steps=100,
        learning_rate=1e-6,
        num_train_epochs=8,
        max_grad_norm=1.0,              # Gradient clipping
        report_to='wandb'
    )
    
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )


    trainer.train()
    
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    artifact = wandb.Artifact(artifact_name, type='model')
    artifact.add_dir(model_dir)
    wandb.log_artifact(artifact)

    wandb.finish()


@app.local_entrypoint()
def main():
    setup_environment()
    wandb_run = wandb.init(project="Baleegh", name=f"BaleeghDPO @ {datetime.now()}")
    # Commit the starting run to enable multi environment runs
    wandb_run.finish()

    train_model.remote(wandb_run.id)


    wandb_run.finish()