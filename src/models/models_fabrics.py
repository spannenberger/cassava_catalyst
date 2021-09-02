from torch import nn
import torch
from models.models_classes import ClassificationModel
from models.models_classes import MetricLearningModel


class ModelsFabric:
    """ A class for creating models for classification tasks with 
    the possibility of extension for other tasks, for example, metric_learning
    """
    
    @classmethod
    def create_model(cls, model, mode, **kwargs):
        if mode == 'Classification':
            return cls.create_classification(model, **kwargs)
        if mode == 'MetricLearning':
            return cls.create_metric_learning(model, **kwargs)

    @staticmethod
    def remove_classifier(model):
        classifier_name, old_classifier = model._modules.popitem()

        if isinstance(old_classifier, nn.Sequential):
            embedding_size = old_classifier[-1].in_features
            old_classifier[-1] = nn.Sequential()

        elif isinstance(old_classifier, nn.Linear):
            embedding_size = old_classifier.in_features
            old_classifier = nn.Sequential()
        else:
            raise Exception("Uknown type of classifier {}".format(type(old_classifier)))
        model.add_module(classifier_name, old_classifier)
        return embedding_size

    @staticmethod
    def create_classification(model, **kwargs):
        path = kwargs.pop('path', '')
        diff_classes_flag = kwargs.pop('diff_classes_flag', False)
        is_local = kwargs.pop('is_local', False)
        old_num_classes = kwargs.pop('old_num_classes', 10)
        num_classes = kwargs.pop('num_classes', 10)
        embedding_size = ModelsFabric.remove_classifier(model)

        if is_local:
            if diff_classes_flag:
                classificator = nn.Linear(embedding_size, old_num_classes)
                model = ClassificationModel(model, classificator, embedding_size)
                model.load_state_dict(torch.load(path)['model_state_dict'])
                classificator = nn.Linear(embedding_size, num_classes)
                model.replace_classifier(classificator)
            else:
                classificator = nn.Linear(embedding_size, num_classes)
                model = ClassificationModel(model, classificator, embedding_size)

                model.load_state_dict(torch.load(path)['model_state_dict'])
        else:
            classificator = nn.Linear(embedding_size, num_classes)
            model = ClassificationModel(model, classificator, embedding_size)

        return model

    @staticmethod
    def create_metric_learning(model, **kwargs):
        path = kwargs.pop('path', '')
        is_local = kwargs.pop('is_local', False)
        embedding_size = ModelsFabric.remove_classifier(model)
        model = MetricLearningModel(model, embedding_size)
        if is_local:
            model.load_state_dict(torch.load(path)['model_state_dict'])
        return model
