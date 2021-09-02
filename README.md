# cassava-catalyst
### This repository contains solution for Cassava-Leaf-Disease Classification based on Catalyst Framework
----
### Content
- [User guide](#user-guide)
  * [Repository Structure](#структура-репозитория)
  * [Instructions for using the repository](#инструкция-по-использованию-репозитория)
- [Информация о конвертации моделей](#информация-о-конвертации-моделей)
- [Информация о моделях](#информация-о-моделях)
- [Training run](#training-run)
# User guide
### Структура репозитория
- [classifications_shells](#training-run) - папка, содержащая скрипты запуска решений задач классификации
- [config](./config) - папка с конфигами эксперимента, в которых мы можем изменять: модель, путь до данных, шедулеры, коллбэки и тд
    * [Multiclass](config/classification/multiclass/train_multiclass.yml) - конфиг мультикласс классификации
- [src](src/) - папка с основными файлами проекта, в которую добавляются новые шедулеры, модели, коллбэки и тд
- [docker-compose.yml](#test-in-docker) - конфиг файл для докера
- [requirements.txt](/requirements.txt) - файл с библиотеками и инструментами, которые нам нужны в проектах
---
### Инструкция по использованию репозитория
- [Multiclass](#запуск-и-изменение-multiclass-решения)
- [Callbacks](#использование-колбэков-в-пайплайне)
 ### Запуск и изменение multiclass решения
   - Склонировать репозиторий
   -  Запустить команду ```pip install -r requirements.txt```
   -  Для изменения, подключения данных обучения:
       - По стандарту данные идут в формате:
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

           test_metadata.csv
           train_metadata.csv
        ```
        - Структура csv файлов
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
       - Изменить в папке ```./config/classification/multiclass/train_multiclass.yml``` файл, прописав новые пути до данных в блоке ```data:```
       - Подготовка данных к эксперименту происходит в ```./src/classification/SupervisedRunner.py``` в методе get_datasets класса ```MulticlassRunner```,
       чтение данных во время эксперимента происходит в ```dataset.py```
   - Изменение моделей обучения
       - Изменить в ```train_multiclass.yml``` файле название модели (доступные модели можно посмотреть в таблице([Информация о моделях](#информация-о-моделях))
   - Логирование эксперимента в mlflow
   - Для отключения колбэков достаточно их закомментировать в config файле
   - Для дообучения на своих моделях:
     - Проверить, что данная модель реализована в пайплайне
     - Создать в корне проекта папку ```our_models```
     - Загрузить в данную папку вашу модель в формате .pth с названием файла "best". Пример: ```best.pth```
     - Пример:
     ```
     model:
        _target_: Densenet121 # имя клаccа. Сам класс будет сконструирован в registry по этому имени
        mode: Classification
        num_classes: &num_classes 2
        path: 'our_models/best.pth' # Путь до расположения вашей локальной модели
        is_local: True # True если обучаете локально загруженную модель
        diff_classes_flag: True # True, если есть разница в кол-ве классов
        old_num_classes: 18 # Если diff_classes_flag=True, то указать кол-во классов в предобученной модели
     ```
### Использование колбэков в пайплайне
- prunning callback прунит параметры во время и/или после обучения.
:neutral_face:**Стоит отметить, что при использовании данного колбэка при обучении multilabel и metric learninng не будет работать конвертация моделей в onnx и torchscript**:neutral_face:
  ```
    Args:
        pruning_fn: функция из torch.nn.utils.prune module.
            Возможные prunning_fn в пайплайне: 'l1_unstructured', 'random_unstructured', 
            'ln_structured', 'random_structured'
            Смотри документацию pytorch: 
            https://pytorch.org/tutorials/intermediate/pruning_tutorial.html
        amount: количество параметров для обрезки.
            Если с плавающей точкой, должно быть от 0,0 до 1,0 и
            представляют собой долю параметров, подлежащих сокращению.
            Если int, он представляет собой абсолютное число
            параметров для обрезки.
        keys_to_prune: Cписок строк. Определяет
            какой тензор в модулях будет обрезан.
        prune_on_epoch_end: флаг bool определяет вызвать или нет
            pruning_fn в конце эпохи.
        prune_on_stage_end: флаг bool определяет вызвать или нет
            pruning_fn в конце стейджа.
        remove_reparametrization_on_stage_end: если True тогда вся
            перепараметризация pre-hooks и tensors с маской
            будет удален в конце стейджа.
        layers_to_prune: список строк - имена модулей, которые нужно удалить.
            Если не указано ни одного, то будет обрезан каждый модуль в
            модели.
        dim: если вы используете structured pruning method вы должны
            указать dimension.
        l_norm: если вы используете ln_structured вы должны указать l_norm.
  ```

- quantization callback квантизирует модель :neutral_face:
  ```
    Args:
      logdir: путь до папки модели после квантизации
      qconfig_spec (Dict, optional): конфиг квантизации в PyTorch формате.
          Defaults to None.
      dtype (Union[str, Optional[torch.dtype]], optional):
          Тип весов после квантизации.
          Defaults to "qint8".
  ```
- CheckpointCallback - сохраняет n лучших моделей
  - best.pth - лучшая модель за обучение
  - last.pth - модель с последней эпохи
  - stage.1_full.pth, ..., stage.n_full.pth - лучшие n моделей за обучение
# Информация о моделях

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