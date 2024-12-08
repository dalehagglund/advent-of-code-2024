from typing import (
    Callable,
)
import time

class timer[T]:
    def __init__(self, clock: Callable[[], T] = time.perf_counter):
        self._laps = []
        self._clock = clock
        self._state = "stopped"
    def _click(self):
        self._laps.append(self._clock())
    def start(self) -> "timer":
        if self._state == "running":
            raise ValueError("timer already running")
        self._state = "running"
        self._laps = []
        self._click()
        return self
    def lap(self):
        if self._state != "running":
            raise ValueError("timer not running")
        self._click()
    def stop(self):
        if self._state == "stopped":
            raise ValueError("timer already stopped")
        self._click()
        self._state = "stopped"
    def elapsed(self):
        if self._state == "stopped":
            return self._laps[-1] - self._laps[0]
        else:
            return self._clock() - self._laps[0]

    def __enter__(self):
        self.start()
        return self
    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.stop()
        # let the exception, if any, happen normally
        return False