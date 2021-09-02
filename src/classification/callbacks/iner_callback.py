from catalyst.dl import Callback, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.registry import Registry
from sklearn.metrics import accuracy_score
from pathlib import Path
from torch.nn import BCELoss, Sigmoid
from torch.nn import CrossEntropyLoss
import torch


class MainInerCallback(Callback):
    """
    InerCallback - кастомный колбэк
    Записывает в файл preds.csv, в нашем случае,
    картинку(ее путь), то что предсказала наша модель,
    то что должна была предсказать и лосс этой картинки(использовалось для
    просмотра фотографий с большим лоссом)
    """

    def __init__(self,
                 subm_file,
                 **kwargs):
        super().__init__(CallbackOrder.Internal)
        self.subm_file = Path(subm_file)
        self.subm_file.parent.mkdir(parents=True, exist_ok=True)
        self.final = []
        self.ac = -1
        self.ac_score_f = accuracy_score

    def on_loader_start(self, _):
        self.paths = []
        self.targets = []
        self.preds = []
        self.losses = []

    def on_loader_end(self, state: IRunner):
        cur_ac = self.ac_score_f(torch.Tensor(self.preds), torch.Tensor(self.targets))
        if state.is_valid_loader:
            if cur_ac > self.ac:
                self.ac = cur_ac
                self.final = []
                for path, pred, target, loss in zip(self.paths, self.preds, self.targets, self.losses):
                    self.final.append((path, pred, target, loss.tolist()))

    def on_experiment_end(self, _):
        subm = ["path;class_id;target;losses"]
        subm += [f"{path};{cls};{tar};{los}" for path, cls, tar, los in self.final]
        with self.subm_file.open(mode='w') as file:
            file.write("\n".join(subm)+"\n")

@Registry
class MulticlassInerCallback(MainInerCallback):

    def __init__(self, subm_file, **kwargs):
        super().__init__(subm_file)
        self.loss_f = CrossEntropyLoss(reduction='none')

    def on_batch_end(self, state: IRunner):
        if state.is_valid_loader:
            self.paths += state.batch["image_name"]
            targets = state.batch['targets'].detach().cpu()
            logits = state.batch["logits"].detach().cpu()
            self.targets += targets.tolist()
            self.losses += list(self.loss_f(logits, targets))
            self.preds += torch.argmax(logits, dim=1).tolist()
