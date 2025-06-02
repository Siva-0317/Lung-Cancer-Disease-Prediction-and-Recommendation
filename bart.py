import torch
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset

# Load the preprocessed dataset
data_files = {"train": "bart_training_data.csv"}
dataset = load_dataset("csv", data_files=data_files)["train"]

# Split into training and evaluation sets
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Load tokenizer and model
model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name,force_download=True)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Tokenization function
def preprocess_function(examples):
    inputs = [ex for ex in examples["input_text"]]
    targets = [ex for ex in examples["target_text"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing
tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_eval = eval_dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./bart_finetuned",
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    learning_rate=5e-5,
    per_device_train_batch_size=2,  # Reduce if memory issues
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    fp16=torch.cuda.is_available(),  # Use FP16 if available
    eval_accumulation_steps=4,  # Helps with memory efficiency
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,  # Add evaluation dataset
    tokenizer=tokenizer,
)

# Start training
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./bart_finetuned_model")
tokenizer.save_pretrained("./bart_finetuned_model")

print("Fine-tuning complete! Model saved to './bart_finetuned_model'")