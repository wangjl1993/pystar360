import logging
import os
import pathlib
import sys
from datetime import datetime
from logging import Formatter
from pathlib import Path
from typing import Union, Tuple
from contextvars import ContextVar
import re


# setting
ENCODING = "UTF-8"
DEFAULT_DATE_FORMAT = "%Y%m%d"

# 日志记录相关的配置
CONSOLE_LOG_LEVEL = logging.DEBUG
FILE_BACKUP_LOG_LEVEL = logging.DEBUG
FILE_ABNORMAL_LOG_LEVEL = logging.ERROR
LOG_IS_DELAY_CREATE = True

# 当前项目的绝对路径
PROJ_PATH = pathlib.Path(__file__).parent.parent.parent
# 日志存储文件夹
LOG_DIR_PATH = PROJ_PATH / "LOG"

SEPARATE_LOG_BY_TASK = True

TASK_ID_CTX_KEY = "task_id"
_task_id_ctx_var: ContextVar[str] = ContextVar(
    TASK_ID_CTX_KEY, default=""
)

def get_log_prefix() -> str:
    """获取 Log 文件的前缀
    前缀的格式为 [channel1,channel2,...]
    将监听的通道和 worker_name 结合起来 

    Returns:
        str: Log 文件的前缀
    """
    start = False
    channel_numbers = []

    for arg in sys.argv[1:]:  # type: str
        if start is False:
            if "--channels" in arg:
                start = True
                continue
        else:
            if arg.isdigit():
                channel_numbers.append(arg)
            else:
                break

    return '[' + "_".join(channel_numbers) + ']'

def get_log_name(log_stem: str = "", prefix: str = "", suffix: str = "", ext: str = ".log") -> str:
    """获取 Log 文件的名字，文件的主干为当前的 worker_name

    Args:
        log_stem (str, optional): 文件主干
        prefix (str, optional): prefix 前缀，放在文件主干的前面
        suffix (str, optional): 可提供一个可选的 suffix 后缀，放在文件主干后面，文件的扩展名之前，
                                用于区分不同用途的日志. Defaults to "".
        ext (str, optional): log 文件的后缀

    Returns:
        str: 完整的 log 文件名
    """
    default_suffix = datetime.now().strftime("%y%m%d")
    suffixes = (default_suffix, suffix)

    if not log_stem:
        try:
            worker_name = sys.argv[1]
        except IndexError:
            worker_name = ""
        log_stem = worker_name

    log_stem = re.sub(r"[^-_\w]", "_", log_stem)
    log_prefix = f"{prefix}{get_log_prefix()}"

    if not ext.startswith("."):
        ext = f".{ext}"

    return f"{log_prefix}" + "_".join((log_stem, *suffixes)) + f"{ext}"



def get_task_id() -> str:
    return _task_id_ctx_var.get()

def set_task_id(val: str):
    return _task_id_ctx_var.set(val)


def get_context_from_task_id(task_id: str) -> Tuple[str, str, str]:
    sep = "_"
    parts = task_id.split(sep)
    train_id = sep.join(parts[:2])
    car_id = sep.join(parts[2:4])
    return train_id, car_id, parts[-1]


class MyFileHandler(logging.FileHandler):
    def __init__(
        self,
        save_dir: Union[str, Path],
        log_stem: str,
        mode="a",
        encoding=ENCODING,
        log_prefix="",
        log_suffix="",
        delay=False,
    ):
        """文件日志处理器

        Args:
            save_dir: (str): 保存的目录
            log_stem (str): 文件名的主干部分
            mode (str, optional): 写入模式，默认是追加写模式. Defaults to 'a'.
            encoding (str, optional): 文件的编码. Defaults to None.
            log_prefix (str, optional): log 文件的前缀
            log_suffix (str, optional): log 文件的后缀
        """
        self.save_dir = save_dir
        self.log_stem = log_stem
        self.log_prefix = log_prefix
        self.log_suffix = log_suffix
        filename = self.get_current_file_name()
        super().__init__(filename, mode=mode, encoding=encoding, delay=delay)

    # noinspection PyBroadException
    # pylint: disable=broad-except
    def emit(self, record):
        """
        Emit a record.
        """
        try:
            if self.should_switch(record):
                self.do_switch()

            if self.stream is None:  # delay was set
                self.stream = self._open()
            logging.FileHandler.emit(self, record)
        except Exception:
            self.handleError(record)

    def get_current_file_name(self) -> Path:
        raise NotImplementedError("")

    # noinspection PyUnusedLocal
    def should_switch(self, record) -> bool:
        raise NotImplementedError("")

    def do_switch(self):
        # close old stream
        self.close_stream()
        # open a new stream
        file_name = self.get_current_file_name()
        self.baseFilename = str(file_name.absolute())
        self.stream = self._open()

    # noinspection PyTypeChecker
    def close_stream(self):
        if self.stream:
            stream = self.stream
            self.stream = None
            if hasattr(stream, "close"):
                stream.close()


class TaskContextFileHandler(MyFileHandler):
    def __init__(
        self,
        save_dir: Union[str, Path],
        log_stem: str,
        mode="a",
        encoding=ENCODING,
        log_prefix="",
        log_suffix="",
        delay=False,
    ):
        """文件日志处理器

        Args:
            save_dir: (str): 保存的目录
            log_stem (str): 文件名的主干部分
            mode (str, optional): 写入模式，默认是追加写模式. Defaults to 'a'.
            encoding (str, optional): 文件的编码. Defaults to None.
            log_prefix (str, optional): log 文件的前缀
            log_suffix (str, optional): log 文件的后缀
        """
        self.current_task_id = self.get_current_task_id()
        super().__init__(save_dir, log_stem, mode, encoding, log_prefix, log_suffix, delay)

    @staticmethod
    def get_current_task_id() -> str:
        return get_task_id()

    def get_current_file_name(self):
        sep = "_"
        task_id = self.current_task_id
        # 有检测任务发过来的时候
        if task_id:
            train_id, car_id, task_id = get_context_from_task_id(task_id)
            channel, carriage = car_id.split(sep)
            date_str = datetime.now().strftime(DEFAULT_DATE_FORMAT)
            task_log_dir = pathlib.Path(self.save_dir) / f"{train_id}_{task_id.upper()}"
            task_log_dir.mkdir(exist_ok=True)
            file_name = (
                pathlib.Path(task_log_dir) /
                f"[{channel:>03}{sep}{carriage:>02}]{date_str}{sep}{self.log_suffix}.log"
            )
            # 按需生成日志文件
            # file_name.touch(exist_ok=True)
            return file_name
        # 防止 task_id 为空的情况, 如还未收到请求前的日志, 或其它意外情况
        file_name = pathlib.Path(self.save_dir) / get_log_name(
            self.log_stem, prefix=self.log_prefix, suffix=self.log_suffix
        )
        return file_name

    # noinspection PyUnusedLocal
    def should_switch(self, record) -> bool:
        """是否需要切换文件名

        Args:
            record (LogRecord): 将要写入日志的记录

        Returns:
            bool: 决定是否要切换
        """
        now_task_id = self.get_current_task_id()
        if now_task_id != self.current_task_id:
            self.current_task_id = now_task_id
            return True
        else:
            return False


class DateSeparateFileHandler(MyFileHandler):
    def __init__(
        self,
        save_dir: Union[str, Path],
        log_stem: str,
        mode="a",
        encoding=ENCODING,
        log_prefix="",
        log_suffix="",
        delay=False,
    ):
        """文件日志处理器

        Args:
            save_dir: (str): 保存的目录
            log_stem (str): 文件名的主干部分
            mode (str, optional): 写入模式，默认是追加写模式. Defaults to 'a'.
            encoding (str, optional): 文件的编码. Defaults to None.
            log_prefix (str, optional): log 文件的前缀
            log_suffix (str, optional): log 文件的后缀
        """
        self.current_day = self.get_current_day()
        super().__init__(save_dir, log_stem, mode, encoding, log_prefix, log_suffix, delay)

    @staticmethod
    def get_current_day() -> int:
        return int(datetime.now().strftime("%y%m%d"))

    def get_current_file_name(self):
        file_name = pathlib.Path(self.save_dir) / get_log_name(
            self.log_stem, prefix=self.log_prefix, suffix=self.log_suffix
        )
        # file_name.touch(exist_ok=True)
        return file_name

    # noinspection PyUnusedLocal
    def should_switch(self, record) -> bool:
        """是否需要切换文件名

        Args:
            record (LogRecord): 将要写入日志的记录

        Returns:
            bool: 决定是否要切换
        """
        now_day = self.get_current_day()
        if now_day > self.current_day:
            self.current_day = now_day
            return True
        else:
            return False


def get_file_handler(save_dir, file_name, *, formatter, log_level, delay=False):
    p_save_dir = pathlib.Path(save_dir)

    if log_level < logging.WARN:
        suffix = "backup"
    else:
        suffix = "abnormal"
        delay = True
    if SEPARATE_LOG_BY_TASK:
        file_hd = TaskContextFileHandler(p_save_dir, file_name, log_suffix=suffix, delay=delay)
    else:
        file_hd = DateSeparateFileHandler(p_save_dir, file_name, log_suffix=suffix, delay=delay)
    file_hd.setLevel(log_level)
    file_hd.setFormatter(formatter)
    return file_hd


def setup_logger(
    name,
    save_dir,
    file_name="log.txt",
    *,
    console_logger_level: int = logging.DEBUG,
    file_backup_logger_level: int = logging.DEBUG,
    file_abnormal_logger_level: int = logging.WARN,
    delay=False,
):
    logger = logging.getLogger(name)
    logger.setLevel(console_logger_level)
    if logger.hasHandlers():
        logger.handlers.clear()

    console_hd = logging.StreamHandler(stream=sys.stdout)
    console_hd.setLevel(console_logger_level)
    _format = "".join(
        [
            "%(levelname)-1.1s %(asctime)-.23s ",
            "[%(filename)s:%(lineno)d(%(funcName)s)]: %(message)s",
        ]
    )
    # 记录 pid
    formatter = Formatter(_format)
    console_hd.setFormatter(formatter)
    logger.addHandler(console_hd)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for level in [
            file_backup_logger_level,
            file_abnormal_logger_level,
        ]:
            file_hd = get_file_handler(
                save_dir, file_name, formatter=formatter, log_level=level, delay=delay
            )
            logger.addHandler(file_hd)

    logger.propagate = False
    return logger


w_logger = setup_logger(
    "worker",
    LOG_DIR_PATH,
    file_name="",  # program generated
    console_logger_level=CONSOLE_LOG_LEVEL,
    file_backup_logger_level=FILE_BACKUP_LOG_LEVEL,
    file_abnormal_logger_level=FILE_ABNORMAL_LOG_LEVEL,
    delay=LOG_IS_DELAY_CREATE,
)
