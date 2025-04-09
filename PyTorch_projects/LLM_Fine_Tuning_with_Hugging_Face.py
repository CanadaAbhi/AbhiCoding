#pip install transformers datasets accelerate wandb
from huggingface_hub import login

login()  # Enter your Hugging Face token to upload models


from datasets import load_dataset

# Load Yelp Reviews dataset
dataset = load_dataset("yelp_review_full")

# Preprocess the dataset (tokenize, pad, and truncate)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

encoded_dataset = dataset.map(preprocess_function, batched=True)

# Split into train and validation subsets
train_dataset = encoded_dataset["train"].shuffle(seed=42).select(range(10000))  # Use a subset for faster training
eval_dataset = encoded_dataset["test"].shuffle(seed=42).select(range(1000))


from transformers import AutoModelForSequenceClassification

# Load BERT with a classification head (num_labels=5 for Yelp Reviews)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)


from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",          # Directory to save checkpoints
    evaluation_strategy="epoch",    # Evaluate at the end of each epoch
    learning_rate=2e-5,             # Learning rate
    per_device_train_batch_size=16, # Batch size for training
    per_device_eval_batch_size=16,  # Batch size for evaluation
    num_train_epochs=3,             # Number of epochs
    weight_decay=0.01,              # Weight decay for regularization
    logging_dir="./logs",           # Directory for logs
    logging_steps=10,
    save_strategy="epoch",
    push_to_hub=False               # Set to True if uploading to Hugging Face Hub
)


from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)


trainer.train()

results = trainer.evaluate()
print(results)

# Save locally
trainer.save_model("./fine_tuned_model")

# Push to Hugging Face Hub (optional)
trainer.push_to_hub("fine-tuned-bert-yelp")

#bash run following:
accelerate config  # Follow prompts to configure multi-GPU or TPU setups

from accelerate import Accelerator

accelerator = Accelerator()

model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss / gradient_accumulation_steps  # Gradient accumulation
        accelerator.backward(loss)

        if step % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    # Evaluation loop here...


#Advanced finetuning
#pip install peft bitsandbytes


from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    task_type="SEQ_CLS",  # Sequence classification task
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()

training_args.fp16 = True  # Enable mixed precision in TrainingArguments

#bash 
#pip install trl

from trl import SFTTrainer

sft_trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
sft_trainer.train()


pip install wandb

import wandb

wandb.init(project="llm-fine-tuning")
training_args.report_to = "wandb"





