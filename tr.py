import os
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    AutoModelForCausalLM,
    TrainerCallback,
)
from trl import SFTTrainer
import wandb
import json


# Login to Weights and Biases

run = wandb.init(project="rewind_article")
# Seed for reproducibility
torch.manual_seed(42)


# Configuration
model_name = "sshleifer/tiny-gpt2"
max_seq_length = 1024
output_dir = "./results"
num_train_epochs = 1
per_device_train_batch_size = 2
per_device_eval_batch_size = 2
gradient_accumulation_steps = 16
learning_rate = 5e-6
logging_steps = 10
save_steps = 10
eval_steps = 10
warmup_steps = 0
save_total_limit = 5  # will save best and latest 
train_file_path = './final_ds/train_completions.jsonl'
val_file_path = './final_ds/test_completions.jsonl'
loss_exploded = False  
explosion_step = 0 


# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def load_jsonl_data(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data


train_data = load_jsonl_data(train_file_path)[:10000]
val_data = load_jsonl_data(val_file_path)[:200]


# Convert data to Hugging Face Dataset format
data_list_train = [dict(d) for d in train_data]
data_list_val = [dict(d) for d in val_data]


train_dataset = Dataset.from_list(data_list_train)
val_dataset = Dataset.from_list(data_list_val)


# Filter examples based on max_seq_length
def filter_examples(example):
    combined_text = example['input']
    tokens = tokenizer.encode(combined_text)
    return len(tokens) < max_seq_length


train_dataset = train_dataset.filter(filter_examples)
val_dataset = val_dataset.filter(filter_examples)


# Format chat template
def format_chat_template(example):
    return {'text': f"\n{example['input']}\n\n{example['model_name']}\n\n{example['output']}\n"}


# Format and prepare datasets
train_dataset = train_dataset.map(format_chat_template)
val_dataset = val_dataset.map(format_chat_template)


print(f"Number of examples in the train set: {len(train_dataset)}")
print(f"Number of examples in the validation set: {len(val_dataset)}")


def create_and_prepare_model():
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


model, tokenizer = create_and_prepare_model()


training_arguments = TrainingArguments(
    num_train_epochs=num_train_epochs,
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    save_total_limit=save_total_limit,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=False,
    bf16=False,
    evaluation_strategy="steps",
    eval_steps=eval_steps,
    warmup_steps=warmup_steps,
    lr_scheduler_type="linear",
    report_to='none',
    save_steps=save_steps,
    save_strategy="steps",
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)


class NoiseInjectionCallback(TrainerCallback):
    def __init__(self, noise_start_step, noise_std):
        self.noise_start_step = noise_start_step
        self.noise_std = noise_std


    def on_step_end(self, args, state, control, **kwargs):
        global loss_exploded
        print(state.global_step)
        if state.global_step == self.noise_start_step and not loss_exploded:
            print(f"Injecting noise to model weights at step {state.global_step}")
            model = kwargs['model']
            with torch.no_grad():
                for param in model.parameters():
                    noise = torch.randn_like(param) * self.noise_std
                    param.add_(noise)


class WandbLoggingCallback(TrainerCallback):
    def __init__(self):
        self.initial_loss = None


    def on_log(self, args, state, control, **kwargs):
        global loss_exploded, explosion_step, run
        logs = kwargs.get('logs', {})
        if 'loss' in logs:
            current_loss = logs['loss']
            run.log({'train_loss': current_loss}, step=state.global_step)
            print("step", state.global_step, "loss", current_loss)
            
            if self.initial_loss is None:
                self.initial_loss = current_loss
            # else:
            if current_loss > 2 * self.initial_loss:  # 200% increase
                print(f"Loss increased by over 200%: {current_loss}. Stopping training.")
                control.should_training_stop = True
                loss_exploded = True 
                explosion_step = int((state.global_step - 2*logging_steps)) # go back 2 steps 
                    


wandb_logging_callback = WandbLoggingCallback()
noise_callback = NoiseInjectionCallback(noise_start_step=40, noise_std=4.0)


def reload_model_and_resume():
    global model, run, explosion_step
    print(f"Explosion step: {explosion_step}")
    rwd_step = str(int(explosion_step))
    # Construct the directory name of the lowest checkpoint
    lowest_checkpoint_dir = f'checkpoint-{rwd_step}'
    model_dir = os.path.join(output_dir, lowest_checkpoint_dir)
    
    print("Loading model from", model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    run = wandb.init(project="rewind_article", resume_from=f"{run.id}?_step={str(int(rwd_step))}")
    
    return model, model_dir




model_pth = ""
limit = 0
while True: 
    limit+=1 
    if loss_exploded: 
        model, model_pth = reload_model_and_resume()
        # loss_exploded = False 
        resume_checkpoint = True
    else:
        resume_checkpoint = False
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        callbacks=[noise_callback, wandb_logging_callback],  # Add both callbacks
        packing=True
    )


    trainer.train(resume_from_checkpoint=False if model_pth == "" else model_pth)
    run.finish()
    if limit == 2:
        break 
