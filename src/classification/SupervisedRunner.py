from catalyst.core import IRunner
from catalyst.runners import SupervisedConfigRunner
from collections import OrderedDict
from pathlib import Path
import pandas as pd
import numpy as np
from dataset import CustomDataset
import torch
from sklearn.model_selection import train_test_split

class MulticlassRunner(IRunner):
    """Кастомный runner нашего эксперимента"""

    def get_datasets(self, stage: str, **kwargs):
        """Работа с данными, формирование train и valid"""
        datasets = OrderedDict()
        data_params = self._stage_config[stage]["data"]
        image_size = data_params.get("image_size")
        if image_size is not None:
            assert len(image_size) == 2
            image_size = tuple(image_size)

        train_dir = Path(data_params["train_dir"])
        metadata_path = train_dir.joinpath(data_params["train_meta"])

        train_images_dir = train_dir.joinpath(data_params["train_image_dir"])

        train_meta = pd.read_csv(metadata_path)

        train_meta["label"] = train_meta["label"].astype(np.int64)

        train_image_paths = [train_images_dir.joinpath(
            i) for i in train_meta["image_path"]]

        train_labels = train_meta["label"].tolist()

        image_paths_train, image_paths_val, \
        labels_train, labels_val = train_test_split(train_image_paths, train_labels,
                                                    stratify=train_labels,
                                                    test_size=data_params["valid_size"])

        datasets["train"] = {'dataset': CustomDataset(image_paths_train,
                                                      labels_train,
                                                      transform_path=data_params['transform_path'])
                             }
        datasets["valid"] = CustomDataset(image_paths_val,
                                          labels_val,
                                          transform_path=data_params['transform_path'],
                                          valid=True)

        return datasets


class MulticlassSupervisedRunner(MulticlassRunner, SupervisedConfigRunner):
    pass