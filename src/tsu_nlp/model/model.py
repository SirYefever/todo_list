import argparse
import json
import shutil
from logging import exception

import optuna #
import pandas as pd #
import sys
import logging
from tsu_nlp.model.logger import SingletonLogger
from clearml import Task, Logger

# Инициализация задачи ClearML
task = Task.init(
    project_name="Spaceship Titanic",
    task_name="CatBoost Model Training",
    tags=["classification", "catboost"]
)

# Инициализация логгера
logger = SingletonLogger().get_logger()
optuna_logger = optuna.logging.get_logger("optuna")
optuna_logger.handlers = logger.handlers


class LoggerWriter:
    """
    Класс для перенаправления вывода в логгер
    """

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():  # Игнорируем пустые строки
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass


# Перенаправляем stdout и stderr в логгер
sys.stdout = LoggerWriter(logger, logging.INFO)
sys.stderr = LoggerWriter(logger, logging.ERROR)


class My_TextNormalization_Model:
    """
    Класс для обучения и нормализации русскоязычного текста.
    """

    def __init__(self):
        """
        Инициализация класса с указанием путей сохранения модели и результатов предсказания
        """
        self.dictionary_path = '../data/dictionary/model_dictionary.json'
        self.results_path = '../data/results.csv'
        self.train_path = '../data/inputs/ru_train.csv'
        self.test_path = '../data/inputs/ru_test_2.csv'
        self.result_path = '../data/results/result.csv'



    def train_dict(self):
        """
        Создание словаря для каждого уникального 'before' с самым часто встречающимся 'after'
        """
        logger.info(f"Начало создания словаря...")

        train = pd.read_csv(self.train_path, encoding='utf-8')
        train['before'] = train['before'].str.lower()
        train['after'] = train['after'].str.lower()
        train['after_c'] = train['after'].map(lambda x: len(str(x).split()))
        train[~(train['class'] == 'LETTERS') & (train['after_c'] > 4)]
        train = train.groupby(['before', 'after'], as_index=False)['sentence_id'].count()
        train = train.sort_values(['sentence_id', 'before'], ascending=[False, True])
        train = train.drop_duplicates(['before'])
        dictionary = {key: value for (key, value) in train[['before', 'after']].values}
        logger.info(f"Словарь инициализирован.")

        try:
            logger.info(f"Попытка записи словаря в файл {self.dictionary_path}")
            with open(self.dictionary_path, 'w') as f:
                json.dump(dictionary, f, indent=4)
        except:
            logger.error(f"Не удалось записать словарь в файл.")

        logger.info(f"Процесс создания словаря завершен успешно.")

    def normalize_text_dict(self):
        """
        Нормализация текста на основе словаря self.dictionary
        """
        logger.info(f"Начало нормализации.")

        try:
            logger.info(f"Попытка считать словарь и файла {self.dictionary_path}")
            with open(self.dictionary_path, 'r') as f:
                dictionary = json.load(f)
        except:
            logger.error(f"Не удалось считать словарь и файла {self.dictionary_path}")

        try:
            logger.info(f"Попытка считать файл с данными {self.test_path}")
            test = pd.read_csv(self.test_path)
        except:
            logger.error(f"Не удалось считать файл с данными {self.test_path}")

        test = pd.read_csv(self.test_path)
        test['id'] = test['sentence_id'].astype(str) + '_' + test['token_id'].astype(str)
        test['before_l'] = test['before'].str.lower()
        test['after'] = test['before_l'].map(
            lambda x: dictionary[x] if x in dictionary else x)

        def fcase(obefore, lbefore, after):
            if lbefore == after:
                return obefore
            else:
                return after

        test['after'] = test.apply(lambda r: fcase(r['before'], r['before_l'], r['after']), axis=1)

        test[['id', 'after']].to_csv(self.result_path, index=False)


if __name__ == '__main__':
    # Указываем основной класс для тренировки модели и предсказания
    classifier = My_TextNormalization_Model()

    # Указываем параметры необходимые к передаче
    parser = argparse.ArgumentParser(description="Обучение и нормализация.")
    parser.add_argument("mode", choices=["train", "normalize"], help="Режим работы: обучение или нормализация.")
    parser.add_argument("method", choices=["dictionary", "neural"], help="Метод обработки: словарем или машинным обучением.")
    # parser.add_argument("--dataset", required=True, help="Полный путь к датасету для обучения или нормализации.")

    # Считываем параметры
    args = parser.parse_args()

    # Если train - обучаем модель, если predict - делаем предсказание
    if args.mode == "train":
        if args.method == "dictionary":
            classifier.train_dict()
    elif args.mode == "normalize":
        if args.method == "dictionary":
            classifier.normalize_text_dict()
