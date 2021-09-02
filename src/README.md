# Contents of src folder
:+1:    :metal: :metal:    :+1:
----

### sct folder structure
- callbacks - folder with shared callbacks
    - convert_callbacks - folder with convertation callbacks
        - onnx_save_callback - converting the model to onnx
        - quantization - model quantization
        - torchscript_save_callback - converting the model to torchscript
    - logger_callbacks - folder with callbacks for logging the experiment, both during the experiment and after
        - mlflow_image_logger - callback with which there is a logging of photos, models, configs in mlflow
        - tensorboard_image_logger - a callback with which photos are logged into tensorboard
- classification - folder with the implementation of solving multiclass classification problems
- models - folder with all models that can be used
    - models_classes - file with model classes for different tasks
    - models_fabrics - model factory file
    - models - classes of different model architectures
- utils - folder with the necessary tools
### Classification folder structure
- callbacks - folder with custom callbacks
- \__init__.py - the file in which we initialize, import the tools we need both from the catalist and our custom
- dataset.py - the file in which we work with the received data from the dataset
- SupervisedRunner.py - Custom runner of our experiment
- transform.py - parser file that applies the specified augmentations to data (light, medium, hard)