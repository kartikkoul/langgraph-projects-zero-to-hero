import logging

import colorlog

# Configure colored console logging: each level gets its own color
# (red for errors/critical) while keeping our message format.
_handler = logging.StreamHandler()
_handler.setFormatter(
    colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)s: %(filename)s(Line: %(lineno)d) - %(asctime)s - %(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )
)

logging.basicConfig(level=logging.DEBUG, handlers=[_handler])

# Silence noisy third-party libraries: only show their WARNING and above.
NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "openai",
    "urllib3",
    "langchain",
    "langsmith",
    "langchain_community",
    "langchain_openai",
    "psycopg",
    "asyncio",
    "watchfiles",
    "onnxruntime",
    "rapidocr",
    "RapidOCR",
    "unstructured",
)


def silence_noisy_loggers() -> None:
    """Lower noisy third-party loggers to WARNING.

    Call this AFTER the libraries are imported.
    """
    for noisy_logger in NOISY_LOGGERS:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


silence_noisy_loggers()