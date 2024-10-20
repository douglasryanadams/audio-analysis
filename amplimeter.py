import queue
import sys

import librosa
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
import sounddevice as sd
from matplotlib.animation import FuncAnimation
from scipy._lib._ccallback import CData
from scipy.signal import decimate

import log

log.setup()

input_device = sd.query_devices(sd.default.device, 'input')
logger.info("Listening to device: {}", input_device['name'])
sample_rate = input_device['default_samplerate'] / 2
logger.info("Sample Rate: {}", sample_rate)

length_milliseconds = 2_000
downsample = 180
frames_per_second = 60
interval = int(1000 / frames_per_second)

data_queue = queue.Queue()
dataset_width = int(length_milliseconds * sample_rate / (1000 * downsample))
data_points = np.zeros(dataset_width)

room_baseline = None
peak_decibels = -80
peak_points = np.array([peak_decibels] * dataset_width)

def audio_callback(indata: np.ndarray, frames: int, time: CData, status: sd.CallbackFlags):
    global room_baseline
    logger.debug("  status: {}", status)
    logger.debug("  length of indata: {}", len(indata[:]))
    # data_queue.put(indata[::downsample, 0])  # For amplitude

    # This copies indata, and prevents very small values from creating weird anomolies
    decimated_amplitudes = decimate(indata[:, 0], downsample)
    positive_amplitudes = np.maximum(np.abs(decimated_amplitudes), 1e-10)
    if room_baseline is None:
        room_baseline = np.mean(positive_amplitudes)
        logger.info("Set room_baseline: {}", room_baseline)

    decibels = librosa.amplitude_to_db(positive_amplitudes, ref=1)
    data_queue.put(decibels)


def update_plot(frame):
    global data_points, peak_decibels, peak_points
    try:
        while True:
            try:
                points = data_queue.get_nowait()
                logger.debug("  length of points: {}", len(points))
            except queue.Empty:
                logger.debug("Queue is empty!")
                break
            percentile_98 = np.percentile(points, 98)
            logger.debug(" 98th percentile: {}", percentile_98)
            if percentile_98 > peak_decibels:
                peak_decibels = percentile_98

            shift_size = len(points)
            data_points = np.roll(data_points, -shift_size)
            data_points[-shift_size:] = points

            peak_points = np.roll(peak_points, -shift_size)
            peak_points[-shift_size:] = np.array([peak_decibels] * shift_size)

        for column, line in enumerate(plot_line):
            line.set_ydata(data_points[:])

        for column, line in enumerate(peak_line):
            line.set_ydata(peak_points[:])

        return plot_line + peak_line
    except KeyboardInterrupt:
        logger.info("Listening complete")
        sys.exit(0)


figure, axes = plt.subplots()
plot_line = axes.plot(data_points, linewidth=0.5)
peak_line = axes.plot(peak_points, linewidth=1, color='r')


# axes.axis((0, len(data_points), -1, 1))
axes.axis((0, len(data_points), -100, 0))
axes.yaxis.grid(True)
axes.tick_params(bottom=False, top=False, labelbottom=False, right=False)
axes.set_xlabel("Time (Milliseconds)")
axes.set_ylabel("Amplitude (dB)")
plt.title("Real-time Microphone Amplitude in Decibels")

figure.tight_layout(pad=2)

# Hardcoding dtype to float32 makes other calculations easier, amplitudes will always be -1 > x > 1
with sd.InputStream(
    device=input_device['index'],
    blocksize=2 ** 10,
    channels=1,
    dtype='float32',
    samplerate=sample_rate,
    callback=audio_callback
):
    animation_ref = FuncAnimation(figure, update_plot, interval=interval, blit=True)
    logger.info("Starting RawStream")
    plt.show()

logger.info("Animation Reference: {}", animation_ref)
