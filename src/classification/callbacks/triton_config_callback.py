from catalyst.dl import Callback, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.registry import Registry
from pathlib import Path
import yaml
import os

@Registry
class TritonConfigCreator(Callback):

    def __init__(self, conf_path='', count=0, kind='', gpus=[], mode=''):

        super().__init__(CallbackOrder.Internal)
        self.conf_path = Path(conf_path)
        self.conf_path.parent.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.count = count
        self.kind = kind
        self.gpus = gpus


    def on_stage_start(self, state: IRunner):

        print(os.path.abspath(state.hparams['args']['configs'][0]))
        with open(os.path.abspath(state.hparams['args']['configs'][0]), encoding="utf-8") as config_yaml:
            params = yaml.safe_load(config_yaml)
            self.aug_path = params['stages']['stage']['data']['transform_path']
            try:
                self.output_size = params['model']['num_classes']
            except KeyError:
                pass
            try:
                self.output_size = params['stages']['stage']['callbacks']['criterion']['embeding_size']
            except KeyError:
                pass
            
        with open(self.aug_path, encoding="utf8") as aug_yaml:
            params = yaml.safe_load(aug_yaml)
            height = params['train']['transforms'][1]['height']
            width = params['train']['transforms'][1]['width']
        
        with open(self.conf_path, "a") as triton_config:
            triton_config.write('platform: "onnxruntime_onnx"\n')
            triton_config.write('input [\n')
            triton_config.write('{\n')
            triton_config.write(f'\tname: "input"\n\tdata_type: TYPE_FP32\n\tdims: [-1, 3, {height}, {width}]\n')
            triton_config.write('}\n]\n')
            triton_config.write('output [\n')
            triton_config.write('{\n')
            triton_config.write(f'\tname: "output"\n\tdata_type: TYPE_FP32\n\tdims: [-1, {self.output_size}]\n')
            triton_config.write('}\n]\n')
            triton_config.write('instance_group [\n')
            triton_config.write('{\n')
            if self.count != 'None':
                triton_config.write(f'\tcount: {self.count}\n\t')
            else: pass
            if self.kind != 'None':
                triton_config.write(f'kind: {self.kind}\n\t')
            else: pass
            if self.gpus != 'None':
                triton_config.write(f'gpus: {self.gpus}\n')
            else: pass
            triton_config.write('}\n')
            triton_config.write(']\n')