import logging
import threading
import inspect
import sys
import os

# ANSI-коды цветов для консольного вывода
class Colors:
    RED = "\033[91m"    # ERROR
    GREEN = "\033[92m"  # INFO
    YELLOW = "\033[93m" # WARNING
    PURPLE = "\033[95m" # CRITICAL
    CYAN = "\033[96m"   # DEBUG
    WHITE = "\033[97m"  # default
    RESET = "\033[0m"   # reset


class SingletonLogger:
    """
    Класс реализует паттерн Singleton для логгера
    """
    _instance = None
    _lock = threading.Lock()
    _logger = None

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(SingletonLogger, cls).__new__(cls)
                    cls._instance._setup_logger()
        return cls._instance

    def _setup_logger(self):
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)

        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)

        # File handler with UTF-8 encoding
        file_handler = logging.FileHandler('logs/normalization.log', encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # Console handler with UTF-8 encoding for Windows
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        if sys.platform == 'win32':
            sys.stdout.reconfigure(encoding='utf-8')

        # Create formatters and add them to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s.%(funcName)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add the handlers to the logger
        self._logger.addHandler(file_handler)
        self._logger.addHandler(console_handler)

    def _get_formatter(self, is_colored):
        """
        Создает и возвращает форматтер для логов.
        """
        if is_colored:
            return self._get_colored_formatter()
        return logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def _get_colored_formatter(self):
        """
        Создает и возвращает цветной форматтер для консольного вывода.
        """

        class ColoredFormatter(logging.Formatter):
            def format(self, record):
                """
                Форматирует запись лога с добавлением цвета и информации о вызывающем коде.
                """
                frame = inspect.stack()[8]  # Индекс для получения вызывающего фрейма
                module = inspect.getmodule(frame[0])
                module_name = module.__name__ if module else ''
                class_name = ''
                if 'self' in frame[0].f_locals:
                    class_name = frame[0].f_locals['self'].__class__.__name__
                function_name = frame[3]
                caller_name = f"{module_name}.{class_name}.{function_name}".strip('.')

                # Применение цвета в зависимости от уровня логирования
                color = Colors.WHITE  # По умолчанию белый
                if record.levelno == logging.DEBUG:
                    color = Colors.CYAN
                elif record.levelno == logging.INFO:
                    color = Colors.GREEN
                elif record.levelno == logging.WARNING:
                    color = Colors.YELLOW
                elif record.levelno == logging.ERROR:
                    color = Colors.RED
                elif record.levelno == logging.CRITICAL:
                    color = Colors.PURPLE

                record.levelname = f"{color}{record.levelname}{Colors.RESET}"
                record.name = caller_name
                return super().format(record)

        return ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def set_level(self, level):
        """
        Устанавливает уровень логирования.
        """
        self._logger.setLevel(level)

    def get_logger(self):
        """
        Возвращает экземпляр логгера.
        """
        return self._logger
