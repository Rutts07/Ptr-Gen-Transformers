# Integrating Transformers with Pointer Generatot Networks

Rahothvarman P \
Nukit Tailor \
Ayan Datta

### Directory Structure :

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



