import modal
import torch
import wandb
import os
import numpy as np
from huggingface_hub import HfApi, HfFolder
import transformers
from tqdm import tqdm
from peft import PeftModel, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
from transformers import DataCollatorForSeq2Seq, BitsAndBytesConfig
from datasets import load_dataset
from evaluate import load
from transformers import get_scheduler
from datetime import datetime

app = modal.App("Training")

image = modal.Image.debian_slim().pip_install([
    "transformers", "torch", "wandb", "evaluate", "huggingface_hub", "datasets",
    "bert_score", "evaluate", "numpy", "peft", "accelerate", "bitsandbytes", "torchvision",
    "nltk", "unbabel-comet"
])

def setup_environment(wandb_run_id=None):
    HfFolder.save_token('REDACTED')
    wandb.login(key='REDACTED')
    seed = 1
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    transformers.set_seed(seed)
    np.random.seed(seed)
    
    if wandb_run_id is not None:
        wandb.init(project="Baleegh", id=wandb_run_id, resume="must")
        print(f"Resuming run: {wandb_run_id}")
    # print(f"GPU is available: {torch.cuda.is_available()}")
    # print(torch.cuda.get_device_name(0))

def setup_lora(model):
    lora_config = LoraConfig(
        init_lora_weights="olora",
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        bias="none",
        r=32, 
        lora_alpha=32, 
        lora_dropout=0.05,
        target_modules=["self_attn.k_proj",
                    "self_attn.q_proj",
                    "self_attn.v_proj",
                    "self_attn.out_proj",
                    "encoder_attn.k_proj",
                    "encoder_attn.q_proj",
                    "encoder_attn.v_proj",
                    "encoder_attn.out_proj",
                    "fc1",
                    "fc2"]
    )

    model = get_peft_model(model, lora_config)
    
    # model.enable_input_require_grads()
    # model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    prepare_model_for_kbit_training(model)
    
    model.gradient_checkpointing_disable()
    model.gradient_checkpointing_kwargs = {"use_reentrant": False}
    
    
    return model

def load_lora_model_and_tokenizer():
    model_dir = "facebook/nllb-200-distilled-600M"
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir,
                                                  quantization_config=quantization_config,
                                                  torch_dtype=torch.float16,
                                                  low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, src_lang="eng_Latn", tgt_lang="arb_Arab")

    try:
        artifact = wandb.use_artifact('NLLBLoRA:latest', type='model')
        lora_dir = artifact.download()
    except Exception:
        import warnings
        lora_dir = None
        warnings.warn("No WandB logged model found. Using Base Model")
        
    if lora_dir is not None:
        model = PeftModel.from_pretrained(model, lora_dir, is_trainable=True)
    else:
        model = setup_lora(model)

    model.config.forced_bos_token_id = 256011

    return model, tokenizer

def load_model_and_tokenizer():
    try:
        artifact = wandb.use_artifact('NLLB:latest', type='model')
        model_dir = artifact.download()
    except Exception as e:
        import warnings
        warnings.warn("No WandB logged model found. Using Base Model")
        model_dir = "facebook/nllb-200-distilled-600M"

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

def load_and_preprocess_data(tokenizer):

    dataset = load_dataset("Abdulmohsena/Classic-Arabic-English-Language-Pairs")['train']
    preprocess_function = lambda examples: tokenizer(
        examples['en'], text_target=examples['ar'], max_length=256, truncation=True, padding=True, return_tensors='pt')
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=['en', 'ar']).shuffle()
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.20)
    return tokenized_dataset

@app.function(gpu="A10G", image=image, timeout=10000)
def train_model(wandb_run_id, train_lora):
    setup_environment(wandb_run_id)
    
    if train_lora:
        model, tokenizer = load_lora_model_and_tokenizer()
        model_dir = "./logs/lora_weights"
        artifact_name = "NLLBLoRA"
    else:
        model, tokenizer = load_model_and_tokenizer()
        model_dir = "./logs/model_weights"
        artifact_name = "NLLB"
    
    BATCH_SIZE = 8
    
    tokenized_dataset = load_and_preprocess_data(tokenizer)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, return_tensors='pt')
    train_dataloader = torch.utils.data.DataLoader(tokenized_dataset["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator, pin_memory=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)
    lr_scheduler = get_scheduler(name="cosine",
                                optimizer=optimizer,
                                num_warmup_steps = round(len(train_dataloader) * 1e-2),
                                num_training_steps = len(train_dataloader))
    
    print("Sanity Check:")
    example = sanity_check(model, tokenizer)[0]
    print(f"Input: {example[0]}")
    print(f"Output: {example[1]}")

    model.train()
    train_loss = 0
    progress_bar = tqdm(train_dataloader)
    
    for step, batch in enumerate(progress_bar, 1):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        
        optimizer.step()
        lr_scheduler.step()
        
        optimizer.zero_grad()
        train_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
        
        if step % 100 == 0:
            wandb.log({"train/train_loss": train_loss / 100})
            print(tokenizer.batch_decode(outputs.logits.argmax(dim=-1), skip_special_tokens=True))
            train_loss = 0

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    artifact = wandb.Artifact(artifact_name, type='model')
    artifact.add_dir(model_dir)
    wandb.log_artifact(artifact)

    wandb.finish()

@app.function(gpu="A10G", image=image, timeout=10000)
def evaluate_model(wandb_run_id, use_lora):
    setup_environment(wandb_run_id)
    BATCH_SIZE = 4
    if use_lora:
        model, tokenizer = load_lora_model_and_tokenizer()
        model = model.merge_and_unload()
    else:
        model, tokenizer = load_model_and_tokenizer()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()
    
    tokenized_dataset = load_and_preprocess_data(tokenizer)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, return_tensors='pt')
    eval_dataloader = torch.utils.data.DataLoader(tokenized_dataset["test"], batch_size=BATCH_SIZE, collate_fn=data_collator, shuffle=True, pin_memory=True)
    
    eval_metrics = {"COMET": 0, "Classicality": 0, "METEOR": 0, "gen_len": 0, "examples": []}
    meteor = load("meteor")
    comet = load("comet")
    fluency = load("Abdulmohsena/classicier")
    
    for step, batch in enumerate(tqdm(eval_dataloader), 1):
        with torch.inference_mode():
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model.generate(**batch)
            references = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
            sources = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            eval_metrics["COMET"] += comet.compute(predictions=predictions, references=references, sources=sources)['mean_score']
            eval_metrics['Classicality'] += fluency.compute(texts=predictions)['classical_score'].mean()
            eval_metrics['METEOR'] += meteor.compute(predictions=predictions, references=references)['meteor']
            eval_metrics["gen_len"] += np.vectorize(len)(np.array(predictions)).mean()
            
            if step % 100 == 0:
                eval_metrics["COMET"] /= 100
                eval_metrics["Classicality"] /= 100
                eval_metrics["METEOR"] /= 100
                eval_metrics["gen_len"] /= 100
                wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()})
                print(eval_metrics)
                eval_metrics = {"COMET": 0, "Classicality": 0, "METEOR": 0, "gen_len": 0, "examples": []}

        torch.cuda.empty_cache()

    examples = sanity_check(model, tokenizer)
    wandb.log({"examples": wandb.Table(columns=["Input", "Output"], data=examples)})
    wandb.finish()

@app.local_entrypoint()
def main():
    setup_environment()
    wandb_run = wandb.init(project="Baleegh", name=f"Baleegh @ {datetime.now()}")
    # Commit the starting run to enable multi environment runs
    wandb_run.finish()

    epochs = 1
    for _ in range(epochs):
        use_lora = False
        # train_model.remote(wandb_run.id, use_lora)
        evaluate_model.local(wandb_run.id, use_lora)

    wandb_run.finish()