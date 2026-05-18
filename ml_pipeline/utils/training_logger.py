from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from time import perf_counter
from typing import Iterable


class TrainingLogger:
    """Logger simple para entrenamientos con formato consistente en consola."""

    def __init__(self, component: str):
        self.component = component
        self._logger = logging.getLogger(f"ml_training.{component}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self._ensure_console_handler()

    def _ensure_console_handler(self) -> None:
        handler_marker = f"_ml_training_handler_{self.component}"
        for handler in self._logger.handlers:
            if getattr(handler, handler_marker, False):
                return

        handler = logging.StreamHandler(sys.stdout)
        setattr(handler, handler_marker, True)
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter(
                fmt=f"[%(asctime)s] [{self.component}] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        self._logger.addHandler(handler)

    def info(self, message: str, *args: object) -> None:
        self._logger.info(message, *args)

    def warning(self, message: str, *args: object) -> None:
        self._logger.warning(message, *args)

    def error(self, message: str, *args: object) -> None:
        self._logger.error(message, *args)

    def exception(self, message: str, *args: object) -> None:
        self._logger.exception(message, *args)


    @contextmanager
    def timed(self, message: str):
        start = perf_counter()
        self.info("%s | inicio", message)
        try:
            yield
        finally:
            elapsed = perf_counter() - start
            self.info("%s | fin en %.2fs", message, elapsed)

    def keras_epoch_callback(
        self,
        *,
        phase: str = "Keras",
        metric_keys: Iterable[str] | None = None,
    ):
        import tensorflow as tf

        parent_logger = self
        phase_name = phase
        selected_metric_keys = list(metric_keys or [])

        class _KerasEpochLogger(tf.keras.callbacks.Callback):
            def on_train_begin(self, logs=None):
                epochs = self.params.get("epochs", "?")
                steps = self.params.get("steps", "?")
                parent_logger.info(
                    "%s | inicio model.fit: epochs=%s, steps_por_epoch=%s",
                    phase_name,
                    epochs,
                    steps,
                )

            def on_epoch_begin(self, epoch, logs=None):
                total_epochs = self.params.get("epochs", "?")
                parent_logger.info("%s | epoch %s/%s iniciado", phase_name, epoch + 1, total_epochs)

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                metrics = self._format_metrics(logs)
                suffix = f" | {metrics}" if metrics else ""
                parent_logger.info("%s | epoch %s finalizado%s", phase_name, epoch + 1, suffix)

            def _format_metrics(self, logs: dict) -> str:
                keys = selected_metric_keys or list(logs.keys())
                parts: list[str] = []
                for key in keys:
                    if key not in logs:
                        continue
                    value = logs[key]
                    try:
                        parts.append(f"{key}={float(value):.4f}")
                    except (TypeError, ValueError):
                        parts.append(f"{key}={value}")
                return ", ".join(parts)

        return _KerasEpochLogger()
