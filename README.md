# cassava-catalyst
### This repository contains solution for Cassava-Leaf-Disease Classification based on Catalyst Framework
----
### Content
- [User guide](#user-guide)
  * [Repository Structure](#repository-structure)
  * [Instructions for using the repository](#instructions-for-using-the-repository)
- [Convertation model info](#convertation-model-info)
- [Model info](#model-info)
- [Training run](#training-run)
- [Docker run](#docker-run)
# User guide
### Repository Structure
- [classifications_shells](#training-run) - folder which contains running scripts for multiclass tasks
- [config](./config) - folder with experiment config in which we can modify: train model, data paths, shedulers, callbacks and etc 
    * [Multiclass](config/classification/multiclass/train_multiclass.yml) - multiclass classification config
- [src](src/) - folder with core files: callbacks, custom runner, loggers and etc
- [docker-compose.yml](#test-in-docker) - config file for docker
- [requirements.txt](/requirements.txt) - file with libs, tools for our repository
---
### Instructions for using the repository
- [Multiclass](#how-to-use-multiclass-solution-for-your-tasks)
- [Callbacks](#how-to-use-callbacks)
 ### How to use multiclass solution for your tasks 
   - Clone this repository
   - Run command ```pip install -r requirements.txt```
   -  How to change data structure and add into pipeline:
       - Default data structure:
       ```
          train_dataset/
            - images/
              - train/
                train_image_name_1.jpg
                train_image_name_2.jpg
                ...
                train_image_name_N.jpg

              - test/
                test_image_name_1.jpg
                test_image_name_2.jpg
                ...
                test_image_name_N.jpg

           train_metadata.csv
        ```
        - csv file structure
        ```
        "image_path":
          train_image_name_1.jpg,
          train_image_name_2.jpg,
          ...
          train_image_name_N.jpg,
        "label":
          1,
          0,
          ...
          1
        ```
       - Change ```./config/classification/multiclass/train_multiclass.yml``` file, adding new data paths in block ```data:```
       - Data setup for our experiment occur ```./src/classification/SupervisedRunner.py``` in ```get_datasets``` method. Data reading occur in ```dataset.py```
   - How to change train model
       - In ```train_multiclass.yml``` file change model name (find them in table([Model info](#model-info))
   - Logging expirement to mlflow:
      - In ```./config/classification/multilabel/train_multilabel.yml``` file add urls and your experiment name (if you want it local do not fill url field)
       ```
       loggers:
            mlflow:
       ```
   - To turn off callback just comment it in config file
   - How to retrain your local model:
     - Check that in your model exists in our pipeline (check model table ([Model info](#model-info)))
     - Create a folder in the root of your project
     - Load your local model into your model folder in .pth formatПример: ```./our_model/best.pth```
     - Example:
     ```
     model:
        _target_: Densenet121 # model class name. The class itself will be constructed in the registry by this name
        mode: Classification
        num_classes: &num_classes 2
        path: 'our_models/best.pth' # path to your local model
        is_local: True # True if your train your local model
        diff_classes_flag: True # True if you want train your model to another class number
        old_num_classes: 18 # If diff_classes_flag=True, add your class number in pre-trained model
     ```
### How to use callbacks 
- prunning callback pruned parametrs while/after training
  ```
    Args:
        pruning_fn: function from torch.nn.utils.prune module
            or your based on BasePruningMethod. Can be string e.g.
            `"l1_unstructured"`. See pytorch docs for more details.
            https://pytorch.org/tutorials/intermediate/pruning_tutorial.html
        amount: quantity of parameters to prune.
            If float, should be between 0.0 and 1.0 and
            represent the fraction of parameters to prune.
            If int, it represents the absolute number
            of parameters to prune.
        keys_to_prune: list of strings. Determines
            which tensor in modules will be pruned.
        prune_on_epoch_end: bool flag determines call or not
            to call pruning_fn on epoch end.
        prune_on_stage_end: bool flag determines call or not
            to call pruning_fn on stage end.
        remove_reparametrization_on_stage_end: if True then all
            reparametrization pre-hooks and tensors with mask
            will be removed on stage end.
        layers_to_prune: list of strings - module names to be pruned.
            If None provided then will try to prune every module in
            model.
        dim: if you are using structured pruning method you need
            to specify dimension.
        l_norm: if you are using ln_structured you need to specify l_norm.
  ```

- quantization callback - quantize model :neutral_face:
  ```
    Args:
      logdir: path to folder for saving
      filename: filename
      qconfig_spec (Dict, optional): quantization config in PyTorch format.
          Defaults to None.
      dtype (Union[str, Optional[torch.dtype]], optional):
          Type of weights after quantization.
          Defaults to "qint8".
  ```
- CheckpointCallback - save ```n``` best models
  - best.pth - best model after training
  - last.pth - model from last epoch
  - stage.1_full.pth, ..., stage.n_full.pth - best n model 
# Model info

| model | onnx  | torchscript | embedding_size |
| :---: | :-: | :-: | :-: |
| ResNet18 | True  | True | 512 |
| ResNet34 | True  | True | 512 |
| ResNet50 | True  | True | 2048 |
| ResNet101 | True  | True | 2048 |
| MobilenetV3Small | False  | True | 576 |
| MobilenetV2 | True  | True | 576 |
| MobilenetV3Large | False  | True | 960 |
| ResNext101_32x8d | True  | True | 2048 |
| ResNext50_32x4d | True  | True | 2048 |
| WideResnet50_2 | True  | True | 2048 |
| WideResnet101_2 | True  | True | 2048 |
| EfficientNetb0 | True  | True | 1280 |
| EfficientNetb3 | True  | True | 1536 |
| EfficientNetb4 | True  | True | 1792 |
| Densenet201 | True  | True | 1920 |
| Densenet169 | True  | True | 1664 |
| Densenet161 | True  | True | 2208 |
| Densenet121 | True  | True | 1024 |

# Training run 
```bash
# To check multiclass pipeline
sh classification_shells/check_multiclass.sh
# To usual multiclass train pipeline
sh classification_shells/train_multiclass.sh

# Run tensorflow for visualisation
tensorboard --logdir=logs/ui # for our pipeline
# Run mlflow 
mlflow ui

```
# Docker run 
```
# build ur project, u need to do this only once
docker-compose build

# run docker ur container
docker-compose up

# shutdown ur container
docker-compose stop
```