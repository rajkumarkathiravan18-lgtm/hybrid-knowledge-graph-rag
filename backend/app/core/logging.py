import logging
import sys

from app.core.config import get_settings


SENSITIVE_LOG_KEYS = {
    "openai_api_key",
    "neo4j_password",
    "langsmith_api_key",
    "authorization",
    "api_key",
    "password",
    "token",
}


class SensitiveDataFilter(logging.Filter):
    """
    Basic logging filter designed to reduce accidental
    secret exposure.

    This is defense-in-depth. Application code must still
    avoid logging secrets directly.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage().lower()

        for sensitive_key in SENSITIVE_LOG_KEYS:
            if sensitive_key in message:
                record.msg = (
                    "[Potentially sensitive log message suppressed]"
                )
                record.args = ()
                break

        return True


def configure_logging() -> None:
    """
    Configure application-wide logging.
    """

    settings = get_settings()

    numeric_level = getattr(
        logging,
        settings.log_level,
        logging.INFO,
    )

    root_logger = logging.getLogger()

    root_logger.setLevel(numeric_level)

    # Prevent duplicate handlers during reload/testing.
    if root_logger.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)

    handler.setLevel(numeric_level)

    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler.setFormatter(formatter)

    handler.addFilter(
        SensitiveDataFilter()
    )

    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named application logger.
    """

    return logging.getLogger(name)