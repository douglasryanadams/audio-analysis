import queue
import sys

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
import sounddevice as sd
from matplotlib.animation import FuncAnimation
from scipy._lib._ccallback import CData

import log

log.setup()

input_device = sd.query_devices(sd.default.device, 'input')
logger.info("Listening to device: {}", input_device['name'])
sample_rate = input_device['default_samplerate']
logger.info("Sample Rate: {}", sample_rate)

length_milliseconds = 2_000
downsample = 90
frames_per_second = 60
interval = int(1000 / frames_per_second)

data_queue = queue.Queue()
dataset_width = int(length_milliseconds * sample_rate / (1000 * downsample))
data_points = np.zeros((dataset_width, 1))


def audio_callback(indata: np.ndarray, frames: int, time: CData, status: sd.CallbackFlags):
    logger.debug("  status: {}", status)
    logger.debug("  length of indata: {}", len(indata[:]))
    data_queue.put(indata[::downsample])  # Copies indata


def update_plot(frame):
    global data_points
    try:
        while True:
            try:
                points = data_queue.get_nowait()
                logger.debug("  length of points: {}", len(points))
            except queue.Empty:
                logger.debug("Queue is empty!")
                break
            shift_size = len(points)
            data_points = np.roll(data_points, -shift_size, axis=0)
            data_points[-shift_size:, :] = points
        for column, line in enumerate(lines):
            line.set_ydata(data_points[:, column])
        return lines
    except KeyboardInterrupt:
        logger.info("Listening complete")
        sys.exit(0)


figure, axes = plt.subplots()
lines = axes.plot(data_points, linewidth=0.5)

bits_granularity = 16  # 8, 16, 32
vertical_bound = 2 ** (bits_granularity - 2)
axes.axis((0, len(data_points), -vertical_bound, vertical_bound))
axes.set_yticks([0])
axes.yaxis.grid(True)
axes.tick_params(bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

figure.tight_layout(pad=0)

with sd.InputStream(device=input_device['index'], channels=1, dtype=f'int{bits_granularity}', samplerate=sample_rate,
                    callback=audio_callback):
    animation_ref = FuncAnimation(figure, update_plot, interval=interval, blit=True)
    logger.info("Starting RawStream")
    plt.show()

logger.info("Animation Reference: {}", animation_ref)
