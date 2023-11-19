from pointer_bart import BARTPointer
from transformers import AutoTokenizer
from datasets import Dataset, load_dataset, DatasetDict
from tqdm import tqdm
from torchmetrics.text.rouge import ROUGEScore

model = BARTPointer.from_pretrained("checkpoints/pointer_bart3/checkpoint-2000")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

dataset = load_dataset("cnn_dailymail", "3.0.0")
test_dataset = Dataset.from_dict(dataset["test"][:100])
dataset = DatasetDict({"test": test_dataset})

rouge = ROUGEScore()
rougelp, rougelr, rougelf = 0, 0, 0
counter = 1
for i in tqdm(range(len(dataset["test"]))):
    if len(dataset["test"][i]["article"]) > 1024:
        continue
    
    counter += 1
    article = dataset["test"][i]["article"]
    highlights = dataset["test"][i]["highlights"]
    
    model.pointer.pgen_list = []
    input_ids = tokenizer(article, return_tensors="pt").input_ids
    outputs = model.generate(input_ids=input_ids, max_length=150)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    rouge_score = rouge(summary, highlights)
    rougelp += rouge_score["rougeL_precision"]
    rougelr += rouge_score["rougeL_recall"]
    rougelf += rouge_score["rougeL_fmeasure"]
    
rougelp /= counter
rougelr /= counter
rougelf /= counter

print(f"ROUGE-L Precision: {rougelp:.2f}")
print(f"ROUGE-L Recall: {rougelr:.2f}")
print(f"ROUGE-L F1: {rougelf:.2f}")