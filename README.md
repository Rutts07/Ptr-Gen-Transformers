# Integrating Transformers with Pointer Generator Networks

Rahothvarman P \
Nukit Tailor \
Ayan Datta

### Project Description :
The task we are tackling is of generating an appropriate short summary for any given news article. Given a news article as the input, the model tries to generate a short summary for that article as the output. This generated summary can be extractive or abstractive meaning that it can have same words from the article or generate new appropriate words respectively.

### Directory Structure :
```
├── Baseline_Pointer_Generator
│   ├── PtrGen+Cov_Networks.ipynb
│   └── PtrGen_Networks.ipynb
├── Baseline_Seq2Seq
│   └── Baseline_Seq2Seq.ipynb
├── Finetuned_BART
│   ├── test_bart.py
│   └── train_bart.py
├── Finetuned_T5
│   ├── test_t5.py
│   └── train_t5.py
├── Inference
│   ├── imports.py
│   ├── inference.py
│   ├── pointer_bart.py
│   └── pointer_t5.py
├── Pointer_BART
│   ├── eval_pointer_bart.py
│   ├── pointer.py
│   ├── pointer_bart.py
│   ├── test_pointer_bart.py
│   └── train_pointer_bart.py
├── Pointer_T5
│   ├── eval_pointer_t5.py
│   ├── pointer.py
│   ├── pointer_t5.py
│   ├── test_pointer_t5.py
│   └── train_pointer_t5.py
├── README.md
└── requirements.txt
```

### Guidlines to clone the project : 
- Clone the repository using ```git clone <link_to_the_repo>```
- Using the Conda Package Manager, create a new environment using ```conda create --name <env_name> --file requirements.txt```

### Instructions to train the models :
- Since, absolute paths to datasets (Inshorts) were used, be sure to update absolute paths in the ```train_<model_name>.py``` files in corresponding folders of the models.
- Run the ```train_<model_name>.py``` under each folder to train the corresponding model.
- The model takes around 5 hours to train on the entire CNN daily mail dataset, while it takes about 40 minutes to train on a truncated version with 50k article-summary pairs.
- T5-small models trains faster than the bart-base counterparts due to lesser number of parameters.
- Once, the models are trained, the trainer saves two checkpoints of the same model, one being the best version considering training and evaluation losses and the other after all training iterations.

### Instructions to test the models :
- Load the best versions of the models by changing the model path variables in the corresponding ```test_<model_name>.py``` file. In addition to this, the input articles can be changed by changing the input text in each file.
- Run the ```test_<model_name>.py``` for inference.



