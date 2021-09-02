from catalyst.dl import Callback, CallbackOrder
from catalyst.registry import Registry
from catalyst.core.runner import IRunner
import numpy as np
import pandas as pd
import ast
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm
from utils.utils import get_from_dict
from pathlib import Path


@Registry
class TensorboardMulticlassLoggingCallback(Callback):

    def __init__(self, logging_image_number, **kwargs):
        self.logging_image_number = logging_image_number
        super().__init__(CallbackOrder.ExternalExtra)

    def on_experiment_end(self, state: IRunner):
        """В конце эксперимента логаем ошибочные фотографии, раскидывая их в N папок,
        которые соответствуют class_names в нашем конфиге
        """

        df = pd.read_csv(get_from_dict(
            state.hparams, 'stages:stage:callbacks:infer:subm_file'), sep=';')

        path_list = [i for i in df[df['class_id'] != df['target']]['path']]
        if(len(df[df['class_id'] != df['target']]) <= self.logging_image_number):
            length = len(df[df['class_id'] != df['target']])
        else:
            length = self.logging_image_number

        class_id = [i for i in df[df['class_id'] != df['target']]['class_id']]
        target = [i for i in df[df['class_id'] != df['target']]['target']]

        try:
            class_names = state.hparams['class_names']
        except KeyError:
            class_names = [x for x in range(
                state.hparams['model']['num_classes'])]
        print('We start logging images to tensorboard... please wait')
        for i in tqdm(range(length)):
            image = ToTensor()(Image.open(f"{path_list[i]}"))
            state.loggers['tensorboard'].loggers['valid'].add_image(
                f"{class_names[target[i]]}/{class_id[i]} - {target[i]} error number {i}.png",
                image)
