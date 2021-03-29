from multiclass.TTASupervisedRunner import TTASupervisedRunner

# Импорт моделей
from multiclass.models.ResNet18 import ResNet18 # converter True
from multiclass.models.ResNet34 import ResNet34
from multiclass.models.ResNet50 import ResNet50
from multiclass.models.ResNet101 import ResNet101
from multiclass.models.EffNetb0 import EffNetb0 # converter onnx: False
from multiclass.models.EffNetb3 import EffNetb3 # converter onnx: False
from multiclass.models.EffNetb6 import EffNetb6 # converter onnx: False
from multiclass.models.densenet121 import densenet121 # converter True
from multiclass.models.densenet169 import densenet169 # converter True
from multiclass.models.densenet201 import densenet201 # converter True
from multiclass.models.densenet161 import densenet161 # converter True
from multiclass.models.resnest50 import resnest50 # False
from multiclass.models.resnext50_32x4d import resnext50_32x4d # converter True 
from multiclass.models.resnext101_32x8d import resnext101_32x8d # converter True 
from multiclass.models.WideResnet50_2 import WideResnet50_2
from multiclass.models.WideResnet101_2 import WideResnet101_2
from multiclass.models.MobilenetV2 import MobilenetV2
from multiclass.models.MobilenetV3Large import MobilenetV3Large
from multiclass.models.MobilenetV3Small import MobilenetV3Small
from multiclass.models.ResNet18_swsl import ResNet18_swsl

# Импорт колбэков
from multiclass.callbacks.iner_callback import InerCallback
from multiclass.callbacks.new_scheduler import CustomScheduler
from multiclass.callbacks.logger_callbacks.mlflow_logging_callback import MLFlowloggingCallback
from multiclass.callbacks.logger_callbacks.tensorboard_image_logger import TensorboardImageCustomLogger

# Импорт инструментов каталиста
from catalyst.registry import Registry
from catalyst.loggers.mlflow import MLflowLogger
from catalyst.loggers.tensorboard import TensorboardLogger
from catalyst.loggers.tensorboard import TensorboardLogger
# from catalyst.callbacks.metrics.confusion_matrix import ConfusionMatrixCallback


# Инициализаця
Registry(TTASupervisedRunner)
Registry(ResNet18_swsl)
# Registry(ConfusionMatrixCallback)
