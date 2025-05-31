import argparse
import json
import shutil
from logging import exception
import os
from pathlib import Path

#import optuna #
import pandas as pd #
import sys
import logging
from tsu_nlp.model.logger import SingletonLogger
#from clearml import Task, Logger
import torch
from transformers import T5ForConditionalGeneration, GPT2Tokenizer
from tqdm import tqdm
import onnxruntime as ort
import numpy as np

# Инициализация задачи ClearML
# task = Task.init(
#     project_name="Spaceship Titanic",
#     task_name="CatBoost Model Training",
#     tags=["classification", "catboost"]
# )

# Инициализация логгера
logger = SingletonLogger().get_logger()
# optuna_logger = optuna.logging.get_logger("optuna")
# optuna_logger.handlers = logger.handlers

# Настройка кэширования модели
os.environ['TRANSFORMERS_CACHE'] = 'models_cache'
MODEL_NAME = "saarus72/russian_text_normalizer"
ONNX_MODEL_PATH = "model_repository/text_normalization/1/model.onnx"
ONNX_ENCODER_PATH = "model_repository/text_normalization/1/encoder_model.onnx"

class LoggerWriter:
    """
    Класс для перенаправления вывода в логгер
    """

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.encoding = 'utf-8'

    def write(self, message):
        if message.strip():
            try:
                if isinstance(message, bytes):
                    message = message.decode(self.encoding)
                self.logger.log(self.level, message.strip())
            except Exception as e:
                self.logger.error(f"Error writing log: {str(e)}")

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
        self.dictionary_path = 'data/dictionary/model_dictionary.json'
        self.results_path = 'data/results.csv'
        self.train_path = 'data/inputs/ru_train.csv'
        self.test_path = 'data/inputs/ru_test_2.csv'
        self.result_path = 'data/results/result.csv'
        
        # Create cache directory
        os.makedirs('models_cache', exist_ok=True)

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
            with open(self.dictionary_path, 'w', encoding='utf-8') as f:
                json.dump(dictionary, f, indent=4, ensure_ascii=False)
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
            with open(self.dictionary_path, 'r', encoding='utf-8') as f:
                dictionary = json.load(f)
        except:
            logger.error(f"Не удалось считать словарь и файла {self.dictionary_path}")
            return None

        try:
            logger.info(f"Попытка считать файл с данными {self.test_path}")
            test = pd.read_csv(self.test_path, encoding='utf-8')
        except:
            logger.error(f"Не удалось считать файл с данными {self.test_path}")
            return None

        test = pd.read_csv(self.test_path, encoding='utf-8')
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
        return test

    def normalize_two(self, test_mode=False):
        """
        Комбинированный метод нормализации, использующий сначала словарь, затем нейронную модель с ONNX
        Args:
            test_mode (bool): If True, only process first 10 items for testing
        """
        logger.info("Начало комбинированной нормализации (словарь + нейронная модель ONNX)...")
        
        # Шаг 1: Попытка нормализации через словарь
        logger.info("Шаг 1: Применение словарной нормализации...")
        dict_results = self.normalize_text_dict()
        if dict_results is None:
            logger.error("Не удалось выполнить словарную нормализацию")
            return
            
        # Определяем, какие токены требуют нейронной нормализации
        needs_neural = []
        for idx, row in dict_results.iterrows():
            # Если после словарной нормализации текст не изменился и содержит цифры или латиницу
            if (row['before'] == row['after'] and 
                any(c.isdigit() or (c.isascii() and c.isalpha()) for c in row['before'])):
                needs_neural.append(idx)
        
        logger.info(f"Найдено {len(needs_neural)} токенов для нейронной нормализации")
        
        if not needs_neural:
            logger.info("Все токены успешно нормализованы словарем")
            dict_results[['id', 'after']].to_csv(self.result_path, index=False, encoding='utf-8')
            return
            
        # Шаг 2: Нейронная нормализация для оставшихся токенов с использованием ONNX
        logger.info("Шаг 2: Применение нейронной нормализации (ONNX) для оставшихся токенов...")
        
        try:
            logger.info("Загрузка модели ONNX и токенизатора...")
            
            # Initialize tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME, cache_dir='models_cache')
            
            # Set up ONNX Runtime session with optimizations
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1
            
            # Create ONNX Runtime session
            providers = ['CPUExecutionProvider']  # Start with CPU provider
            try:
                if 'CUDAExecutionProvider' in ort.get_available_providers():
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    logger.info("CUDA доступен. Используется GPU для ONNX Runtime")
                else:
                    logger.warning("CUDA недоступен. Используется CPU для ONNX Runtime")
            except Exception as e:
                logger.warning(f"Ошибка при проверке CUDA: {str(e)}. Используется CPU.")
                
            # Load encoder and decoder ONNX models
            encoder_session = ort.InferenceSession(
                ONNX_ENCODER_PATH,
                providers=providers,
                sess_options=sess_options
            )
            decoder_session = ort.InferenceSession(
                ONNX_MODEL_PATH,
                providers=providers,
                sess_options=sess_options
            )
            
            # Get model input names
            encoder_input_names = [input.name for input in encoder_session.get_inputs()]
            decoder_input_names = [input.name for input in decoder_session.get_inputs()]
            logger.info(f"Доступные входы энкодера: {encoder_input_names}")
            logger.info(f"Доступные входы декодера: {decoder_input_names}")
            
            # Process only tokens that need neural normalization
            neural_texts = dict_results.iloc[needs_neural]
            batch_size = 32  # Reduced batch size for testing
            if test_mode:
                batch_size = 2
                neural_texts = neural_texts.head(10)  # Process only 10 items in test mode
            
            logger.info(f"Размер батча: {batch_size}")
            normalized_neural = []
            
            for i in tqdm(range(0, len(neural_texts), batch_size)):
                batch_texts = neural_texts['before'].iloc[i:i + batch_size].tolist()
                
                formatted_texts = []
                max_len = 0
                for text in batch_texts:
                    if text.isdigit():
                        text_rev = text[::-1]
                        groups = [text_rev[i:i+3][::-1] for i in range(0, len(text_rev), 3)]
                        text = ' '.join(groups[::-1])
                    formatted_text = f"<SC1>[{text}]<extra_id_0>"
                    formatted_texts.append(formatted_text)
                    max_len = max(max_len, len(formatted_text))
                
                # Tokenize inputs
                inputs = tokenizer(
                    formatted_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=min(max_len + 10, 128),
                    return_tensors="np"  # Return numpy arrays for ONNX
                )
                
                # Run encoder
                encoder_inputs = {
                    'input_ids': inputs['input_ids'].astype(np.int64),
                    'attention_mask': inputs['attention_mask'].astype(np.int64)
                }
                encoder_outputs = encoder_session.run(None, encoder_inputs)
                
                # Initialize decoder inputs
                decoder_input_ids = np.array([[tokenizer.pad_token_id]] * len(batch_texts), dtype=np.int64)
                
                # Run decoder with encoder outputs
                decoder_inputs = {
                    'input_ids': decoder_input_ids,
                    'encoder_hidden_states': encoder_outputs[0],
                    'encoder_attention_mask': inputs['attention_mask'].astype(np.int64)
                }
                
                # Generate sequence
                max_length = min(max_len + 20, 128)
                output_ids = [decoder_input_ids]
                
                for _ in range(max_length):
                    decoder_outputs = decoder_session.run(None, decoder_inputs)
                    next_token_logits = decoder_outputs[0][:, -1, :]
                    next_tokens = np.argmax(next_token_logits, axis=-1)
                    
                    # Add the predicted tokens to the sequence
                    next_tokens = next_tokens.reshape(-1, 1)
                    decoder_input_ids = np.concatenate([decoder_input_ids, next_tokens], axis=1)
                    
                    # Update decoder inputs for next iteration
                    decoder_inputs['input_ids'] = decoder_input_ids
                    
                    # Check if all sequences have generated EOS token
                    if all(tokenizer.eos_token_id in seq for seq in decoder_input_ids):
                        break
                
                # Decode outputs
                decoded_outputs = tokenizer.batch_decode(decoder_input_ids, skip_special_tokens=True)
                
                cleaned_outputs = []
                for output in decoded_outputs:
                    text = output.replace("<SC1>", "").replace("<extra_id_0>", "").strip()
                    text = text.strip('[]')
                    cleaned_outputs.append(text)
                
                normalized_neural.extend(cleaned_outputs)
            
            # Update results with neural normalization
            for idx, neural_text in zip(needs_neural[:len(normalized_neural)], normalized_neural):
                dict_results.at[idx, 'after'] = neural_text
            
            # Save final results
            output_path = self.result_path.replace('.csv', '_test.csv') if test_mode else self.result_path
            logger.info(f"Сохранение результатов в {output_path}")
            dict_results[['id', 'after']].to_csv(output_path, index=False, encoding='utf-8')
            logger.info("Комбинированная нормализация успешно завершена")

        except Exception as e:
            logger.error(f"Ошибка при нейронной нормализации: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return

    def normalize_text(self, test_mode=False):
        """
        Нормализация текста с использованием предобученной T5 модели
        Args:
            test_mode (bool): If True, only process first 10 items for testing
        """
        logger.info("Начало нормализации с использованием T5 модели...")

        try:
            logger.info("Загрузка тестового датасета...")
            test = pd.read_csv(self.test_path, encoding='utf-8')
            test['id'] = test['sentence_id'].astype(str) + '_' + test['token_id'].astype(str)
            
            if test_mode:
                logger.info("Запуск в тестовом режиме - обработка только первых 10 элементов")
                logger.info("Примеры входных данных:")
                for _, row in test.head(10).iterrows():
                    logger.info(f"ID: {row['id']}, Before: {row['before']}")
                test = test.head(10)
                
        except Exception as e:
            logger.error(f"Не удалось загрузить тестовый датасет: {str(e)}")
            return

        try:
            logger.info("Загрузка модели и токенизатора...")
            
            # Check if model is already cached
            cache_dir = Path('models_cache')
            if not (cache_dir / 'model').exists():
                logger.info("Загрузка модели из Hugging Face...")
            else:
                logger.info("Использование кэшированной модели...")

            # Check CUDA availability
            if torch.cuda.is_available():
                logger.info(f"CUDA доступен. Используется GPU: {torch.cuda.get_device_name(0)}")
                device = torch.device("cuda")
                # Set memory optimization flags
                torch.cuda.empty_cache()
                torch.backends.cudnn.benchmark = True
                # Enable mixed precision
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            else:
                logger.warning("CUDA недоступен. Используется CPU. Это может занять очень много времени.")
                device = torch.device("cpu")
                
            tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_dir)
            model = T5ForConditionalGeneration.from_pretrained(
                MODEL_NAME, 
                cache_dir=cache_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
                use_cache=True
            )
            
            model = model.to(device)
            model.eval()
            
            logger.info(f"Модель загружена и работает на {device}")
        except Exception as e:
            logger.error(f"Не удалось загрузить модель: {str(e)}")
            return

        normalized_texts = []
        # Optimize batch size for GPU memory
        batch_size = 256 if torch.cuda.is_available() else 32
        if test_mode:
            batch_size = 2
            
        logger.info(f"Размер батча: {batch_size}")

        try:
            logger.info("Начало процесса нормализации...")
            
            # Process in chunks to avoid memory issues
            chunk_size = 10000
            for chunk_start in range(0, len(test), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(test))
                chunk = test[chunk_start:chunk_end]
                
                with torch.inference_mode():  # Faster than no_grad()
                    for i in tqdm(range(0, len(chunk), batch_size)):
                        batch_texts = chunk['before'].iloc[i:i + batch_size].tolist()
                        
                        # Format inputs according to model requirements
                        formatted_texts = []
                        max_len = 0  # For dynamic padding
                        for text in batch_texts:
                            if any(c.isdigit() or c.isascii() and c.isalpha() for c in text):
                                if text.isdigit():
                                    text_rev = text[::-1]
                                    groups = [text_rev[i:i+3][::-1] for i in range(0, len(text_rev), 3)]
                                    text = ' '.join(groups[::-1])
                                formatted_text = f"<SC1>[{text}]<extra_id_0>"
                            else:
                                formatted_text = text
                            formatted_texts.append(formatted_text)
                            max_len = max(max_len, len(formatted_text))
                        
                        # Токенизация с динамическим паддингом
                        inputs = tokenizer(
                            formatted_texts, 
                            padding=True, 
                            truncation=True, 
                            max_length=min(max_len + 10, 128),  # Dynamic max length with small buffer
                            return_tensors="pt"
                        )
                        input_ids = inputs["input_ids"].to(device)
                        attention_mask = inputs["attention_mask"].to(device)

                        # Генерация с оптимизированными параметрами
                        outputs = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_length=min(max_len + 20, 128),  # Dynamic max length for outputs
                            num_beams=2,  # Reduced from 4 for speed
                            early_stopping=True,
                            do_sample=False,  # Deterministic generation is faster
                            use_cache=True,
                            eos_token_id=tokenizer.eos_token_id
                        )

                        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                        
                        cleaned_outputs = []
                        for output, original_text in zip(decoded_outputs, batch_texts):
                            text = output.replace("<SC1>", "").replace("<extra_id_0>", "").strip()
                            text = text.strip('[]')
                            if not any(c.isdigit() or c.isascii() and c.isalpha() for c in original_text):
                                text = original_text
                            cleaned_outputs.append(text)
                        
                        normalized_texts.extend(cleaned_outputs)
                        
                # Clear GPU memory after each chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Создание DataFrame с результатами
            results_df = pd.DataFrame({
                'id': test['id'],
                'after': normalized_texts
            })

            # Сохранение результатов
            output_path = self.result_path.replace('.csv', '_test.csv') if test_mode else self.result_path
            logger.info(f"Сохранение результатов в {output_path}")
            results_df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info("Нормализация успешно завершена")

        except Exception as e:
            logger.error(f"Ошибка при нормализации текста: {str(e)}")
            return


if __name__ == '__main__':
    # Указываем основной класс для тренировки модели и предсказания
    classifier = My_TextNormalization_Model()

    # Указываем параметры необходимые к передаче
    parser = argparse.ArgumentParser(description="Обучение и нормализация.")
    parser.add_argument("mode", choices=["train", "normalize"], help="Режим работы: обучение или нормализация.")
    parser.add_argument("method", choices=["dictionary", "neural", "two"], help="Метод обработки: словарем, машинным обучением или комбинированный.")
    parser.add_argument("--test", action="store_true", help="Запустить в тестовом режиме (только 10 элементов)")
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
        elif args.method == "neural":
            classifier.normalize_text(test_mode=args.test)
        elif args.method == "two":
            classifier.normalize_two(test_mode=args.test)