from SupervisedRunner import MulticlassSupervisedRunner
from SupervisedRunner import MultilabelSupervisedRunner

# Импорт моделей
from models.models import ResNet18
from models.models import MobilenetV3Small
from models.models import MobilenetV3Large
from models.models import MobilenetV2
from models.models import EfficientNetb4
from models.models import EfficientNetb3
from models.models import EfficientNetb0
from models.models import Densenet201
from models.models import Densenet169
from models.models import Densenet161
from models.models import Densenet121
from models.models import ResNet34
from models.models import ResNet50
from models.models import ResNet101
from models.models import ResNext50_32x4d
from models.models import ResNext101_32x8d
from models.models import WideResnet50_2
from models.models import WideResnet101_2

# Импорт колбэков
from callbacks.custom_scheduler import CustomScheduler
from callbacks.convert_callbacks.torchscript_save_callback import TorchscriptSaveCallback
from callbacks.convert_callbacks.onnx_save_callback import OnnxSaveCallback
# Multiclass
from callbacks.iner_callback import MulticlassInerCallback
from callbacks.logger_callbacks.mlflow_image_logger import MLFlowMulticlassLoggingCallback
from callbacks.logger_callbacks.tensorboard_image_logger import TensorboardMulticlassLoggingCallback

from callbacks.logger_callbacks.mlflow_logger import CustomMLflowLogger
# Импорт инструментов каталиста
from catalyst.registry import Registry
from catalyst.loggers.mlflow import MLflowLogger
from catalyst.loggers.tensorboard import TensorboardLogger


# Инициализаця
Registry(MulticlassSupervisedRunner)
Registry(MultilabelSupervisedRunner)
Registry(ResNet18)
Registry(ResNet34)
Registry(ResNet50)
Registry(ResNet101)
Registry(EfficientNetb0)
Registry(EfficientNetb3)
Registry(EfficientNetb4)
Registry(Densenet121)
Registry(Densenet169)
Registry(Densenet201)
Registry(Densenet161)
Registry(ResNext50_32x4d)
Registry(ResNext101_32x8d)
Registry(WideResnet50_2)
Registry(WideResnet101_2)
Registry(MobilenetV2)
Registry(MobilenetV3Large)
Registry(MobilenetV3Small)
Registry(CustomMLflowLogger)
