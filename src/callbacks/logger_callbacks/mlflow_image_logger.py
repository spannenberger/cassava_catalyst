from catalyst.dl import Callback, CallbackOrder
from catalyst.registry import Registry
from catalyst.core.runner import IRunner
import mlflow
import pandas as pd
import ast
from PIL import Image
import numpy as np
from tqdm import tqdm
from utils.utils import get_from_dict
from pathlib import Path


class MainMLFlowLoggerCallback(Callback):

    def __init__(self):
        super().__init__(CallbackOrder.ExternalExtra)

    def on_stage_start(self, state: IRunner):
        """Логаем конфиг эксперимента и аугментации как артефакт в начале стейджа"""
        mlflow.log_artifact(get_from_dict(state.hparams, 'args:configs')[0], 'config')
        mlflow.log_artifact(
            get_from_dict(state.hparams, 'stages:stage:data:transform_path'), 'config/aug_config')

    def on_experiment_end(self, state: IRunner):
        callbacks_dict = get_from_dict(state.hparams, 'stages:stage:callbacks')
        if 'quantization' in callbacks_dict:
            mlflow.log_artifact('logs/quantized.pth', 'quantized_model')
        else:
            print('\nNo such file quantized.pth, because quantization callback is disabled')

        onnx_checkpoint_names = get_from_dict(
            callbacks_dict, 'onnx_saver:checkpoint_names', default=[])
        torchsript_checkpoint_names = get_from_dict(
            callbacks_dict, 'torchscript_saver:checkpoint_names', default=[])

        print('\nStarting logging convert models... please wait')
        print('\nTorchsript:')
        if len(torchsript_checkpoint_names) > 0:
            for model in tqdm(torchsript_checkpoint_names):
                try:
                    path = Path(state.logdir) / get_from_dict(callbacks_dict,
                                                              'torchscript_saver:out_dir') / f'{model}.pt'
                    mlflow.log_artifact(path, 'torchscript_models')
                except FileNotFoundError:
                    print(f'\nNo such file {model}.pt, nothing to log...')
        else:
            print("Torchsript convert callback is disabled\n")
        print('\nOnnx:')
        if len(torchsript_checkpoint_names) > 0:
            for model in tqdm(onnx_checkpoint_names):
                try:
                    path = Path(state.logdir) / get_from_dict(callbacks_dict,
                                                              'onnx_saver:out_dir') / f'{model}.onnx'
                    mlflow.log_artifact(path, 'onnx_models')
                except FileNotFoundError:
                    print(f'\nNo such file {model}.onnx, nothing to log...\n')
        else:
            print("Onnx convert callback is disabled\n")

        if 'prunning' in callbacks_dict:
            path = get_from_dict(callbacks_dict, 'saver:logdir')
            mlflow.log_artifact(f'{path}/last.pth', 'prunned_models')
            mlflow.log_artifact(f'{path}/best.pth', 'prunned_models')
        else:
            print('\nNo prunned models to log\n')

        mlflow.pytorch.log_model(state.model, artifact_path=get_from_dict(
            state.hparams, 'model:_target_'))
        mlflow.end_run()


@Registry
class MLFlowMulticlassLoggingCallback(MainMLFlowLoggerCallback):

    def __init__(self, logging_image_number, **kwargs):
        self.logging_image_number = logging_image_number
        super().__init__()

    def on_experiment_end(self, state: IRunner):
        """В конце эксперимента логаем ошибочные фотографии, раскидывая их в N папок,
        которые соответствуют class_names в нашем конфиге
        """

        df = pd.read_csv(get_from_dict(state.hparams, 'stages:stage:callbacks:infer:subm_file'), sep=';')
        path_list = [i for i in df[df['class_id'] != df['target']]['path']]
        if(len(df[df['class_id'] != df['target']]) <= self.logging_image_number):
            length = len(df[df['class_id'] != df['target']])
        else:
            length = self.logging_image_number
        class_id = [i for i in df[df['class_id'] != df['target']]['class_id']]
        target = [i for i in df[df['class_id'] != df['target']]['target']]
        try:
            class_names = get_from_dict(state.hparams, 'class_names')
        except KeyError:
            class_names = [x for x in range(
                get_from_dict(state.hparams, 'model:num_classes'))]
        print('Start logging images to mlflow... please wait')
        for i in tqdm(range(length)):
            image = Image.open(f"{path_list[i]}")
            mlflow.log_image(
                image,
                f"{class_names[target[i]]}/{class_id[i]} - {target[i]} error number {i}.png"
            )

        super().on_experiment_end(state)
