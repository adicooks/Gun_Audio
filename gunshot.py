import pyaudio
import librosa
import logging
import time
import schedule
import scipy.signal
import numpy as np
import tensorflow as tf
import six
import tensorflow.keras as keras
import soundfile as sf
from threading import Thread, Event
from datetime import timedelta as td
from queue import Queue
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import backend as K
from twilio.rest import Client

# output logger
logger = logging.getLogger('debugger')
logger.setLevel(logging.DEBUG)
ch = logging.FileHandler('output.log')
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

AUDIO_FORMAT = pyaudio.paFloat32
AUDIO_RATE = 44100
NUMBER_OF_AUDIO_CHANNELS = 1
AUDIO_DEVICE_INDEX = 0
NUMBER_OF_FRAMES_PER_BUFFER = 4410
SAMPLE_DURATION = 2
AUDIO_VOLUME_THRESHOLD = 0.01
NOISE_REDUCTION_ENABLED = False
MODEL_CONFIDENCE_THRESHOLD = 0.5
MINIMUM_FREQUENCY = 20
MAXIMUM_FREQUENCY = AUDIO_RATE // 2
NUMBER_OF_MELS = 128
NUMBER_OF_FFTS = NUMBER_OF_MELS * 20
SMS_ALERTS_ENABLED = True
ALERT_MESSAGE = "ALERTNOW: A Gunshot Was Detected on "
NETWORK_COVERAGE_TIMEOUT = 3600
DESIGNATED_ALERT_RECIPIENTS = ["+16096823000"]
SCHEDULED_LOG_FILE_TRUNCATION_TIME = "00:00"
TWILIO_ACCOUNT_SID = 'AC57c41037491ba686c115364e1ce0daaf'
TWILIO_AUTH_TOKEN = 'b361b4b32b3c43c1cfbff43331d7b876'
TWILIO_PHONE_NUMBER = '+18447304187'

sound_data = np.zeros(0, dtype="float32")
noise_sample_captured = False
gunshot_sound_counter = 1
noise_sample = []
audio_analysis_queue = Queue()
sms_alert_queue = Queue()

# binarizing labels
labels = np.load("/Users/adi/Downloads/Gun_Audio/misc/augmented_labels.npy")

labels = np.array([("gun_shot" if label == "gun_shot" else "other") for label in labels])
label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(labels)
labels = np.hstack((labels, 1 - labels))

## Librosa Wrapper Function Definitions ##
def _stft(y, n_fft, hop_length, win_length):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

def _istft(y, hop_length, win_length):
    return librosa.istft(y, hop_length, win_length)

def _amp_to_db(x):
    return librosa.core.logamplitude(x, ref_power=1.0, amin=1e-20, top_db=80.0)

def _db_to_amp(x):
    return librosa.core.perceptual_weighting(x, frequencies=1.0)

# noise reduction function
def remove_noise(audio_clip,
                 noise_clip,
                 n_grad_freq=2,
                 n_grad_time=4,
                 n_fft=2048,
                 win_length=2048,
                 hop_length=512,
                 n_std_thresh=1.5,
                 prop_decrease=1.0,
                 verbose=False):
    """ Removes noise from audio based upon a clip containing only noise

    Args:
        audio_clip (array): The first parameter.
        noise_clip (array): The second parameter.
        n_grad_freq (int): how many frequency channels to smooth over with the mask.
        n_grad_time (int): how many time channels to smooth over with the mask.
        n_fft (int): number audio of frames between STFT columns.
        win_length (int): Each frame of audio is windowed by `window()`. The window will be of length `win_length` and then padded with zeros to match `n_fft`..
        hop_length (int):number audio of frames between STFT columns.
        n_std_thresh (int): how many standard deviations louder than the mean dB of the noise (at each frequency level) to be considered signal
        prop_decrease (float): To what extent should you decrease noise (1 = all, 0 = none)
        verbose: Whether to display time statistics for the noise reduction process

    Returns:
        array: The recovered signal with noise subtracted

    """

    # Debugging
    if verbose:
        start = time.time()

    noise_stft = _stft(noise_clip, n_fft, hop_length, win_length)
    noise_stft_db = _amp_to_db(np.abs(noise_stft))  # Converts the sample units to dB

    # Calculates statistics over the noise sample
    mean_freq_noise = np.mean(noise_stft_db, axis=1)
    std_freq_noise = np.std(noise_stft_db, axis=1)
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh

    if verbose:
        print("STFT on noise:", td(seconds=time.time() - start))
        start = time.time()

    # Takes a STFT over the signal sample
    sig_stft = _stft(audio_clip, n_fft, hop_length, win_length)
    sig_stft_db = _amp_to_db(np.abs(sig_stft))

    if verbose:
        print("STFT on signal:", td(seconds=time.time() - start))
        start = time.time()

    # Calculates value to which to mask dB
    mask_gain_dB = np.min(_amp_to_db(np.abs(sig_stft)))

    if verbose:
        print("Noise Threshold & Mask Gain in dB: ", noise_thresh, mask_gain_dB)

    # Creates a smoothing filter for the mask in time and frequency
    smoothing_filter = np.outer(
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                np.linspace(1, 0, n_grad_freq + 2),
            ]
        )[1:-1],
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                np.linspace(1, 0, n_grad_time + 2),
            ]
        )[1:-1]
    )

    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
    db_thresh = np.repeat(np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
                          np.shape(sig_stft_db)[1],
                          axis=0).T

    sig_mask = sig_stft_db < db_thresh

    if verbose:
        print("Masking:", td(seconds=time.time() - start))
        start = time.time()

    sig_mask = scipy.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
    sig_mask = sig_mask * prop_decrease

    if verbose:
        print("Mask convolution:", td(seconds=time.time() - start))
        start = time.time()

    sig_stft_db_masked = (sig_stft_db * (1 - sig_mask)
                          + np.ones(np.shape(mask_gain_dB))
                          * mask_gain_dB * sig_mask)

    sig_imag_masked = np.imag(sig_stft) * (1 - sig_mask)
    sig_stft_amp = (_db_to_amp(sig_stft_db_masked) * np.sign(sig_stft)) + (1j * sig_imag_masked)

    if verbose:
        print("Mask application:", td(seconds=time.time() - start))

    recovered_signal = _istft(sig_stft_amp, hop_length, win_length)
    recovered_spec = _amp_to_db(
        np.abs(_stft(recovered_signal, n_fft, hop_length, win_length))
    )

    if verbose:
        print("Signal recovery:", td(seconds=time.time() - start))

    return recovered_signal


# Converting 1D Sound Arrays into Spectrograms #
def power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0):
    S = np.asarray(S)
    if amin <= 0:
        logger.debug("ParameterError: amin must be strictly positive")
    if np.issubdtype(S.dtype, np.complexfloating):
        logger.debug("Warning: power_to_db was called on complex input so phase information will be discarded.")
        magnitude = np.abs(S)
    else:
        magnitude = S
    if six.callable(ref):
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)
    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))
    if top_db is not None:
        if top_db < 0:
            logger.debug("ParameterError: top_db must be non-negative")
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)
    return log_spec

def convert_audio_to_spectrogram(data):
    spectrogram = librosa.feature.melspectrogram(y=data, sr=AUDIO_RATE,
                                                 hop_length=HOP_LENGTH,
                                                 fmin=MINIMUM_FREQUENCY,
                                                 fmax=MAXIMUM_FREQUENCY,
                                                 n_mels=NUMBER_OF_MELS,
                                                 n_fft=NUMBER_OF_FFTS)
    spectrogram = power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram


# WAV File Composition Function

# Saves a two-second gunshot sample as a WAV file
def create_gunshot_wav_file(microphone_data, index, timestamp):
    sf.write("/Users/adi/Downloads/Gun_Audio/gunshots/Gunshot Sound Sample #"
             + str(index) + " (" + str(timestamp) + ").wav", microphone_data, 22050)

def clear_log_file():
    with open("output.log", 'w'):
        pass

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

# load model & set input shape
interpreter_1 = tf.lite.Interpreter(model_path = "/Users/adi/Downloads/Gun_Audio/1D.tflite")
interpreter_1.allocate_tensors()
input_details_1 = interpreter_1.get_input_details()
output_details_1 = interpreter_1.get_output_details()
input_shape_1 = input_details_1[0]['shape']

def print_status(stop_event):
    while not stop_event.is_set():
        print("AlertNow is running...")
        time.sleep(15)

### --- ###

# Multithreaded Inference: A callback thread adds two-second samples of microphone data to an audio analysis
# queue; the main thread, an audio analysis thread, detects the presence of gunshot sounds in samples retrieved from
# the audio analysis queue; and an SMS alert thread dispatches groups of messages to designated recipients.

### --- ###

# sms alert thread
def send_sms_alert():
    if SMS_ALERTS_ENABLED:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

        while True:
            sms_alert_status = sms_alert_queue.get()
            sms_alert_timestamp = sms_alert_queue.get()
            if sms_alert_status == "Gunshot Detected":
                try:
                    for number in DESIGNATED_ALERT_RECIPIENTS:
                        message = client.messages.create(
                            body=ALERT_MESSAGE + sms_alert_timestamp,
                            from_=TWILIO_PHONE_NUMBER,
                            to=number
                        )
                    logger.debug(" *** Sent out an SMS alert to all designated recipients using Twilio *** ")
                except Exception as e:
                    logger.debug(f"ERROR: Unable to successfully send an SMS alert using Twilio. Error: {str(e)}")
                    pass
                finally:
                    logger.debug(" ** Finished evaluating an audio sample with the model ** ")

    else:
        while True:
            sms_alert_status = sms_alert_queue.get()
            sms_alert_timestamp = sms_alert_queue.get()
            if sms_alert_status == "Gunshot Detected":
                logger.debug(ALERT_MESSAGE + sms_alert_timestamp)


# sms alert thread
sms_alert_thread = Thread(target=send_sms_alert)
sms_alert_thread.start()
status_stop_event = Event()
status_thread = Thread(target=print_status, args=(status_stop_event,))
status_thread.start()

# Callback Thread
def callback(in_data, frame_count, time_info, status):
    global sound_data
    sound_buffer = np.frombuffer(in_data, dtype="float32")
    sound_data = np.append(sound_data, sound_buffer)
    if len(sound_data) >= 88200:
        audio_analysis_queue.put(sound_data)
        current_time = time.ctime(time.time())
        audio_analysis_queue.put(current_time)
        sound_data = np.zeros(0, dtype="float32")
    return sound_buffer, pyaudio.paContinue


pa = pyaudio.PyAudio()
stream = pa.open(format=AUDIO_FORMAT,
                 rate=AUDIO_RATE,
                 channels=NUMBER_OF_AUDIO_CHANNELS,
                 input_device_index=AUDIO_DEVICE_INDEX,
                 frames_per_buffer=NUMBER_OF_FRAMES_PER_BUFFER,
                 input=True,
                 stream_callback=callback)

stream.start_stream()
logger.debug("--- Listening to Audio Stream ---")
schedule.every().day.at(SCHEDULED_LOG_FILE_TRUNCATION_TIME).do(clear_log_file)

try:
    while True:
        schedule.run_pending()
        microphone_data = np.array(audio_analysis_queue.get(), dtype="float32")
        time_of_sample_occurrence = audio_analysis_queue.get()

        # process & log
        maximum_frequency_value = np.max(microphone_data)
        if maximum_frequency_value >= AUDIO_VOLUME_THRESHOLD:
            modified_microphone_data = librosa.resample(y=microphone_data, orig_sr=AUDIO_RATE, target_sr=22050)
            if NOISE_REDUCTION_ENABLED and noise_sample_captured:
                modified_microphone_data = remove_noise(audio_clip=modified_microphone_data, noise_clip=noise_sample)
                number_of_missing_hertz = 44100 - len(modified_microphone_data)
                modified_microphone_data = np.array(modified_microphone_data.tolist() + [0 for i in range(number_of_missing_hertz)], dtype="float32")
            modified_microphone_data = modified_microphone_data[:44100]

            processed_data_1 = modified_microphone_data
            processed_data_1 = processed_data_1.reshape(input_shape_1)

            interpreter_1.set_tensor(input_details_1[0]['index'], processed_data_1)
            interpreter_1.invoke()
            probabilities_1 = interpreter_1.get_tensor(output_details_1[0]['index'])

            gunshot_probability = probabilities_1[0][1] * 100
            other_probability = probabilities_1[0][0] * 100
            logger.debug(f"Noise classified as: {label_binarizer.inverse_transform(probabilities_1[:, 0])[0]} | Gunshot: {probabilities_1[0][1] * 100:6.2f}% | Other: {probabilities_1[0][0] * 100:6.2f}%")

            model_1_activated = probabilities_1[0][1] >= MODEL_CONFIDENCE_THRESHOLD
            if model_1_activated:
                sms_alert_queue.put("Gunshot Detected")
                sms_alert_queue.put(time_of_sample_occurrence)
                create_gunshot_wav_file(modified_microphone_data, gunshot_sound_counter, time_of_sample_occurrence)
                gunshot_sound_counter += 1

            if probabilities_1[0][1] >= MODEL_CONFIDENCE_THRESHOLD:
                print(f"Gunshot detected with {gunshot_probability:.2f}% Probability! A SMS has been sent to {DESIGNATED_ALERT_RECIPIENTS[0]}")

        elif NOISE_REDUCTION_ENABLED and not noise_sample_captured:
            noise_sample = librosa.resample(y=microphone_data, orig_sr=AUDIO_RATE, target_sr=22050)
            noise_sample = noise_sample[:44100]
            noise_sample_captured = True

except KeyboardInterrupt:
    status_stop_event.set()
    status_thread.join()
