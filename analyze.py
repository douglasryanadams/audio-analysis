import sys
import logging
import inspect

import librosa
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
from scipy.signal import decimate

logger.remove()
logger.add(sys.stdout, level='INFO')

class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())
logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)


logger.info("Loading file")
waveform, sample_rate = librosa.load('sample.wav', duration=60)
logger.info(".. file loaded: {}, {}", waveform, sample_rate)

logger.info("Calculating amplitude in decibels")
decimated_waveform = decimate(waveform, 100)
time = np.linspace(0, len(waveform) / sample_rate, num=len(decimated_waveform))
decibels = np.abs(decimated_waveform)
db_amplitude = librosa.amplitude_to_db(decibels, ref=np.max)
percentile_80 = np.percentile(db_amplitude, 80)
logger.info(".. amplitude calculated")


logger.info("Drawing graph")
plt.figure(figsize=(20, 4))
plt.plot(time, db_amplitude, color='#81D8D0', linewidth=0.5)
plt.axhline(y=percentile_80, color='#D88189', linestyle='-', label=f'({percentile_80:.2f} dB)')

plt.title("Decibels of Audio File")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (dB)")
plt.tight_layout()
plt.savefig("audio-visualized.png")

logger.info(".. graph drawn")

