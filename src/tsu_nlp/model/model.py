import argparse
import shutil
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
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
    Класс для обучения и предсказания с использованием модели CatBoostClassifier.
    """

    def __init__(self):
        """
        Инициализация класса с указанием путей сохранения модели и результатов предсказания
        """
        #self.model_path = 
        self.results_path = './data/results.txt'


    def train(self, dataset_path):
        """
        """

    def normalize_text(self, dataset_path):
        """
        """


if __name__ == '__main__':
    # Указываем основной класс для тренировки модели и предсказания
    classifier = My_TextNormalization_Model()

    # Указываем параметры необходимые к передаче
    parser = argparse.ArgumentParser(description="Обучение и нормализация.")
    parser.add_argument("mode", choices=["train", "normalize"], help="Режим работы: обучение или нормализация.")
    parser.add_argument("--dataset", required=True, help="Полный путь к датасету для обучения или нормализации.")

    # Считываем параметры
    args = parser.parse_args()

    # Если train - обучаем модель, если predict - делаем предсказание
    if args.mode == "train":
        classifier.train(args.dataset)
    elif args.mode == "normalize":
        classifier.normalize_text(args.dataset)
