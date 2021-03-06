from catalyst.dl import Callback, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.registry import Registry
import torch
from pathlib import Path
from tqdm import tqdm


@Registry
class TorchscriptSaveCallback(Callback):

    def __init__(self, out_dir, checkpoint_names):
        super().__init__(CallbackOrder.Internal)
        self.out_dir = Path(out_dir)
        self.checkpoint_names = checkpoint_names

    def on_experiment_end(self, state: IRunner):
        print('\nWe start converting to torchscript... please wait')
        try:
            device = 'cpu:0'
            if torch.cuda.is_available():
                device = 'cuda:0'
            for checkpoint in tqdm(self.checkpoint_names):
                state.model.load_state_dict(torch.load(
                    Path(state.logdir)/'checkpoints'/(checkpoint+'.pth'))['model_state_dict'])
                state.model.eval()
                state.model.to(device)
                x = torch.rand(1, 3, 512, 512)
                x.to(device)
                scripted = torch.jit.script(state.model, x)
                output = state.logdir/self.out_dir / (checkpoint + '.pt')
                output.parent.mkdir(parents=True, exist_ok=True)
                torch.jit.save(scripted, str(output))
        except RuntimeError:
            print('Sorry, we cant convert this model to torchscript...\n')
