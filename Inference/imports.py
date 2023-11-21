from pointer_t5 import T5Pointer
from pointer_bart import BARTPointer
from transformers import AutoTokenizer, T5ForConditionalGeneration, BartForConditionalGeneration

def generate_summary(model_name, article):
    if model_name[:3] == "T5-":
        article = "summarize: " + article
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        
        if model_name == "T5-Ptr-Gen":
            model = T5Pointer.from_pretrained("models/t5-ptr")
            model.pointer.pgen_list = []
            
        elif model_name == "T5-Inshorts":
            model = T5ForConditionalGeneration.from_pretrained("models/t5-ins")
            
        else:
            model = T5ForConditionalGeneration.from_pretrained("models/t5-cnn")
        
    else:
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
        
        if model_name == "Bart-Ptr-Gen":
            model = BARTPointer.from_pretrained("models/bart-ptr")
            model.pointer.pgen_list = []
            
        elif model_name == "Bart-Inshorts":
            model = BartForConditionalGeneration.from_pretrained("models/bart-ins")
            
        else:
            model = BartForConditionalGeneration.from_pretrained("models/bart-cnn")
        
    inputs  = tokenizer.encode(article, return_tensors="pt", max_length=100, truncation=True)
    outputs = model.generate(input_ids=inputs, max_length=150)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return summary

def generate_pgen_scores(model_name, article):
    if model_name[:3] == "T5-":
        article = "summarize: " + article
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        model = T5Pointer.from_pretrained("models/t5-ptr")
        
    else:
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
        model = BARTPointer.from_pretrained("models/bart-ptr")
            
    model.pointer.pgen_list = []
    inputs  = tokenizer.encode(article, return_tensors="pt", max_length=100, truncation=True)
    outputs = model.generate(input_ids=inputs, max_length=150, output_scores=True)
    tokens  = tokenizer.convert_ids_to_tokens(outputs[0])
    for token in tokens:
        tokens[tokens.index(token)] = token.replace("Ġ", "")
    
    pgens = [0] + model.pointer.pgen_list
    
    return tokens, pgens
