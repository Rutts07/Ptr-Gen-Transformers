from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer
from pointer_bart import BARTPointer

dataset = load_dataset("csv", data_files={"train":["Data/inshort_news_data-1.csv", "Data/inshort_news_data-2.csv", "Data/inshort_news_data-3.csv", "Data/inshort_news_data-4.csv", "Data/inshort_news_data-6.csv", "Data/inshort_news_data-7.csv"], "validation":["Data/inshort_news_data-5.csv"]}, delimiter=",", column_names=["news_headline", "news_article"], skiprows=1)
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

max_input_length = 256
max_target_length = 128

def preprocess_examples(examples):
    articles = examples["news_article"]
    highlights = examples["news_headline"]

    inputs = [article for article in articles]
    model_inputs = tokenizer(inputs, max_length=max_input_length, padding="max_length", truncation=True)

    labels = tokenizer(highlights, max_length=max_target_length, padding="max_length", truncation=True).input_ids

    labels_with_ignore_index = []
    for labels_examples in labels:
        labels_examples = [label if label != tokenizer.pad_token_id else -100 for label in labels_examples]
        labels_with_ignore_index.append(labels_examples)
    
    model_inputs["labels"] = labels_with_ignore_index
    return model_inputs

dataset = dataset.map(preprocess_examples, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

model = BARTPointer.from_pretrained("facebook/bart-base")
args = TrainingArguments(output_dir="checkpoints/pointer_bart3", evaluation_strategy="steps", eval_steps=500, per_device_train_batch_size=16, per_device_eval_batch_size=16, save_strategy="steps", save_total_limit=2, load_best_model_at_end=True, num_train_epochs=3, learning_rate=1e-5, remove_unused_columns=False,label_names=["labels"])

trainer = Trainer(model=model, args=args, train_dataset=dataset["train"], eval_dataset=dataset["validation"])
trainer.train()
