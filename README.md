# tsar_pipeline
Данный репозиторий содержит решение мультикласс и мультилейбл классификации
----
### Содержание
- [User guide](#user-guide)
  * [Структура репозитория](#структура-репозитория)
  * [Инструкция по использованию репозитория](#инструкция-по-использованию-репозитория)
- [Использование тритона после обучения](#использование-тритона-после-обучения)
- [Информация о конвертации моделей](#информация-о-конвертации-моделей)
- [Training run](#training-run)
- [Train in docker](#train-in-docker)
# User guide
### Структура репозитория
- [classifications_shells](#training-run) - папка, содержащая скрипты запуска репозитория
- [config](./config) - папка с конфигами эксперимента, в которых мы можем изменять: модель, путь до данных, шедулеры, коллбэки и тд
    * [Multiclass](config/classification/multiclass/train_multiclass.yml) - конфиг мультикласс классификации
    * [Multilabel](config/classification/multilabel/train_multilabel.yml) - конфиг мультилейбл класификаци
- [src](src/) - папка с основными файлами проекта, в которую добавляются новые шедулеры, модели, коллбэки и тд
- [docker-compose.yml](#test-in-docker) - конфиг файл для докера
- [model_converter.py](/model_converter.py) - файл для конвертации моделей в форматы torchscript, onnx и проверки корректности преобразованных файлов
- [requirements.txt](/requirements.txt) - файл с библиотеками и инструментами, которые нам нужны в проектах
---
### Инструкция по использованию репозитория
- [Multiclass](#запуск-и-изменение-multiclass-решения)
- [Multilabel](#запуск-и-изменение-multilabel-решения)
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
       - Изменить в папке ```./config/multiclass/train_multiclass.yml``` файл, прописав новые пути до данных в блоке ```data:```
   - Изменение моделей обучения
       - Изменить в ```train_multiclass.yml``` файле название модели (доступные модели можно посмотреть в ```src/classification/__init__.py``` в ```Registry(some_model)```) в блоке ```model:```
   - Логирование эксперимента в mlflow
   - Для отключения колбэков достаточно их закомментировать в config файле
    - Для дообучения на своих моделях:
       - Создать .env файл
       ```
       USER=YourWorkEmail@napoleonit.ru
       MLFLOW_TRACKING_URI=
       MLFLOW_S3_ENDPOINT_URL=
       AWS_ACCESS_KEY_ID=
       AWS_SECRET_ACCESS_KEY=
       ```
       - Изменить в папке ```./config/multiclass/train_multiclass.yml``` файл, прописав новые url и название эксперимента в блоке 
       ```
       loggers:
            mlflow:```
   - Для отключения колбэков достаточно их закомментировать в config файле
   - Для дообучения на своих моделях:
     - Проверить, что данная модель реализована в пайплайне
     - Создать в корне проекта папку ```our_models```
     - Загрузить в данную папку вашу модель в формате .pth с названием файла "best". Пример: ```best.pth```
     - Пример:
     ```
     model:
        _target_: Densenet121 # имя клаccа. Сам класс будет сконструирован в registry по этому имени
        num_classes: &num_classes 2
        path: 'our_models/best.pth' # Путь до расположения вашей локальной модели
        is_local: True # True если обучаете локально загруженную модель
        diff_classes_flag: True # True, если есть разница в кол-ве классов
        old_num_classes: 18 # Если diff_classes_flag=True, то указать кол-во классов в предобученной модели
     ```
 ### Запуск и изменение multilabel решения
   - Склонировать репозиторий
   - Запустить команду ```pip install -r requirements.txt```
   - Для изменения, подключения данных обучения:
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
        "class_0":
          1.0,
          0.0,
          ...
          1.0
        "class_1":
          1.0,
          0.0,
          ...
          1.0
        "class_2":
          1.0,
          0.0,
          ...
          1.0
        ```
       - Изменить в папке ```./config/multilabel/train_multilabel.yml``` файл, прописав новые пути до данных в блоке ```data:```
       - Подготовка данных к эксперименту происходит в ```./src/multilabel/TTASupervisedRunner.py``` в методе get_datasets,
       чтение данных во время эксперимента происходит в ```dataset.py```
   - Изменение моделей обучения
       - Изменить в ```train_multilabel.yml``` файле название модели (доступные модели можно посмотреть в ```src/classification/__init__.py``` в ```Registry(some_model)```) в блоке ```model:```
   - Логирование эксперимента в mlflow
       - Создать .env файл
       ```
       USER=YourWorkEmail@napoleonit.ru
       MLFLOW_TRACKING_URI=
       MLFLOW_S3_ENDPOINT_URL=
       AWS_ACCESS_KEY_ID=
       AWS_SECRET_ACCESS_KEY=
       ```
       - Изменить в папке ```./config/multiclass/train_multilabel.yml``` файл, прописав новые url и название эксперимента в блоке 
       ```
       loggers:
            mlflow:
       ```
   - Для отключения колбэков достаточно их закомментировать в config файле
   - Для дообучения на своих моделях:
     - Проверить, что данная модель реализована в пайплайне
     - Создать в корне проекта папку ```our_models```
     - Загрузить в данную папку вашу модель в формате .pth с названием файла "best". Пример: ```best.pth```
     - Поставть True в нужных пунктах конфига
     - Пример:
     ```
     model:
        _target_: Densenet121 # имя клаccа. Сам класс будет сконструирован в registry по этому имени
        num_classes: &num_classes 18
        path: 'our_models/best.pth' # Путь до расположения вашей локальной модели
        is_local: True # True если обучаете локально загруженную модель
        diff_classes_flag: True # True, если есть разница в кол-ве классов
        old_num_classes: 2 # Если diff_classes_flag=True, то указать кол-во классов в предобученной модели
     ```
# Использование тритона после обучения
В репозитории реализован колбэк для создания минимального тритон конфига модели, который автоматически логируется в mlflow:
```
triton:
  _target_: TritonConfigCreator
  conf_path: "./logs/triton/config.pbtxt" # Путь создания конфига
  mode: multilabel # Выбрать один из режимов multiclass, multilabel
  # instance_group params
  count: 1 # Кол-во используемых gpu
  kind: None # Читай доку тритона
  gpus: [ 0 ] # номера используемых gpu
```
Также в таблице([Информация о конвертации моделей](#информация-о-конвертации-моделей)) можно посмотреть возможность использования модели в тритоне
# Информация о конвертации моделей     
| model | onnx  | torchscript  | Triton |
| :---: | :-: | :-: | :-: |
| Densenet121 | True  | True  | True  |
| Densenet161 | True  | True  | True  |
| Densenet169 | True  | True  | True  |
| Densenet201 | True  | True  | True  |
| EfficientNetb0 | True  | True  | True  |
| EfficientNetb3 | True  | True  | True  |
| EfficientNetb4 | True  | True  | True  |
| MobilenetV2 | True  | True  | True  |
| MobilenetV3Large | False  | True  | False  |
| MobilenetV3Small | False  | True  | False  |
| ResNet18_swsl | True  | True  | True  |
| ResNet18 | True  | True  | True  |
| ResNet34 | True  | True  | True  |
| ResNet50 | True  | True  | True  |
| ResNet101 | True  | True  | True  |
| Resnext50_32x4d | True  | True  | True  |
| Resnext101_32x8d | True  | True  | True  |
| WideResnet50_2 | True  | True  | True  |
| WideResnet101_2 | True  | True  | True  |
# Training run 
```bash
# To check multiclass pipeline
sh classification_shells/check_multiclass.sh
# To usual multiclass train pipeline
sh classification_shells/train_multiclass.sh


# To check multilabel pipeline
sh classification_shells/check_multilabel.sh
# To usual multilabel train pipeline
sh classification_shells/train_multilabel.sh


# Run tensorflow for visualisation
tensorboard --logdir=logs/ui # for our pipeline
# Run mlflow 
mlflow ui

```
# Train in docker
```
# запуск multilabel решения
docker-compose up -d --build multilabel

# запуск multiclass решения
docker-compose up -d --build multiclass
```