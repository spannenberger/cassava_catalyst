import yaml
import albumentations as A


class CustomAugmentator():
    """
    Парсер yml файлов аугментаций
    """

    def transforms(self, path, aug_mode='train'):
        """
        Чтение yml файлов с аугментациями
        path: путь до yml файла с аугментациями
        aug_mode: для использования отдельных аугментаций для 'train' и 'valid'
        """
        self.path = path
        self.aug_mode = aug_mode
        with open(self.path, encoding="utf8") as f:
            params = yaml.safe_load(f)
        return self.parse(params[self.aug_mode]['transforms'], 'Compose')

    def parse(self, yml_augs_list, mode='Compose'):
        """
        yml_augs_list: список аугментаций в yml файлах
        mode: "Compose" для парсинга transform
              "OneOf" для парсинга transform в -OneOf:
        """
        augs_list = []
        for i in yml_augs_list:
            try:
                transform_name = i['transform']
                params = {j: i[j] for j in i.keys() - {'transform'}}
                augs_list.append(getattr(A, transform_name)(**params))
            except KeyError as e:
                if str(e) != '\'transform\'':
                    raise e
                augs_list.append(self.parse(i['OneOf'], 'OneOf'))
        if mode == 'Compose':
            return A.Compose(augs_list)
        elif mode == 'OneOf':
            return A.OneOf(augs_list)
        raise Exception('Unknown mode')
