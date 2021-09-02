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


class MultilabelRunner(IRunner):
    """Кастомный runner нашего эксперимента"""

    def _run_batch(self):
        self._run_event("on_batch_start")
        self.handle_batch(batch=self.batch)

        self.batch['for_metrics'] = (
            self.batch['logits'] > self.hparams['args']['threshold']).type(torch.ByteTensor)
        self._run_event("on_batch_end")

    def get_datasets(self, stage: str, **kwargs):
        """Работа с данными, формирование train и valid"""
        datasets = OrderedDict()
        data_params = self._stage_config[stage]["data"]
        image_size = data_params.get("image_size")
        if image_size is not None:
            assert len(image_size) == 2
            image_size = tuple(image_size)

        train_dir = Path(data_params["train_dir"])
        train_path = train_dir.joinpath(data_params["train_meta"])
        test_path = train_dir.joinpath(data_params["test_meta"])

        train_images_dir = train_dir.joinpath(data_params["train_image_dir"])
        test_images_dir = train_dir.joinpath(data_params["test_image_dir"])

        train_meta = pd.read_csv(train_path)
        test_meta = pd.read_csv(test_path)
        train_meta['label'] = [i[0] for i in zip(train_meta.iloc[:, 1:].values)]
        test_meta['label'] = [i[0] for i in zip(test_meta.iloc[:, 1:].values)]

        train_image_paths = [train_images_dir.joinpath(i) for i in train_meta["path"]]
        test_image_paths = [test_images_dir.joinpath(i) for i in test_meta["path"]]

        train_labels = train_meta["label"].tolist()
        train_labels = torch.Tensor(train_labels)
        train_labels = train_labels.type(torch.DoubleTensor)

        test_labels = test_meta["label"].tolist()
        test_labels = torch.Tensor(test_labels)
        test_labels = test_labels.type(torch.DoubleTensor)

        image_paths_train = train_image_paths
        labels_train = train_labels
        image_paths_val = test_image_paths
        labels_val = test_labels

        datasets["train"] = {'dataset': CustomDataset(image_paths_train,
                                                      labels_train,
                                                      transform_path=data_params['transform_path']
                                                      )
                             }
        datasets["valid"] = CustomDataset(image_paths_val,
                                          labels_val,
                                          transform_path=data_params['transform_path'],
                                          valid=True)
        return datasets


class MultilabelSupervisedRunner(MultilabelRunner, SupervisedConfigRunner):
    pass
