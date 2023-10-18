# imports
import math
import wave
import struct
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import urllib.request
import tarfile
from audiolm_pytorch import SoundStream, SoundStreamTrainer


"""
# define all dataset paths, checkpoints, etc
dataset_folder = "placeholder_dataset"
soundstream_ckpt = "results/soundstream.8.pt" # this can change depending on number of steps
hubert_ckpt = 'hubert/hubert_base_ls960.pt'
hubert_quantizer = f'hubert/hubert_base_ls960_L9_km500.bin' # listed in row "HuBERT Base (~95M params)", column Quantizer


# Placeholder data generation
def get_sinewave(freq=440.0, duration_ms=200, volume=1.0, sample_rate=44100.0):
    # code adapted from https://stackoverflow.com/a/33913403
    audio = []
    num_samples = duration_ms * (sample_rate / 1000.0)
    for x in range(int(num_samples)):
        audio.append(volume * math.sin(2 * math.pi * freq * (x / sample_rate)))
    return audio

def save_wav(file_name, audio, sample_rate=44100.0):
    # Open up a wav file
    wav_file=wave.open(file_name,"w")
    # wav params
    nchannels = 1
    sampwidth = 2
    # 44100 is the industry standard sample rate - CD quality.  If you need to
    # save on file size you can adjust it downwards. The stanard for low quality
    # is 8000 or 8kHz.
    nframes = len(audio)
    comptype = "NONE"
    compname = "not compressed"
    wav_file.setparams((nchannels, sampwidth, sample_rate, nframes, comptype, compname))
    # WAV files here are using short, 16 bit, signed integers for the
    # sample size.  So we multiply the floating point data we have by 32767, the
    # maximum value for a short integer.  NOTE: It is theortically possible to
    # use the floating point -1.0 to 1.0 data directly in a WAV file but not
    # obvious how to do that using the wave module in python.
    for sample in audio:
        wav_file.writeframes(struct.pack('h', int( sample * 32767.0 )))
    wav_file.close()
    return

def make_placeholder_dataset():
    print(os.path.abspath(dataset_folder))
    # Make a placeholder dataset with a few .wav files that you can "train" on, just to verify things work e2e
    if os.path.isdir(dataset_folder):
        return
    os.makedirs(dataset_folder)
    save_wav(f"{dataset_folder}/example.wav", get_sinewave())
    save_wav(f"{dataset_folder}/example2.wav", get_sinewave(duration_ms=500))
    os.makedirs(f"{dataset_folder}/subdirectory")
    save_wav(f"{dataset_folder}/subdirectory/example.wav", get_sinewave(freq=330.0))

make_placeholder_dataset()"""

# Get actual dataset. Uncomment this if you want to try training on real data

# full dataset: https://www.openslr.org/12
# We'll use https://us.openslr.org/resources/12/dev-clean.tar.gz development set, "clean" speech.
# We *should* train on, well, training, but this is just to demo running things end-to-end at all so I just picked a small clean set.

"""url = "https://us.openslr.org/resources/12/dev-clean.tar.gz"
#filename = "train-clean-100"
filename = "dev-clean"
dataset_folder = filename
filename_targz = filename + ".tar.gz"
if not os.path.isfile(filename_targz):
    urllib.request.urlretrieve(url, filename_targz)
if not os.path.isdir(filename):
    # open file
    with tarfile.open(filename_targz) as t:
        t.extractall(filename)
"""

soundstream = SoundStream(
    codebook_size = 1024,
    rq_num_quantizers = 8,
)

trainer = SoundStreamTrainer(
    soundstream,
    folder = 'train-clean-100',
    batch_size = 4,
    grad_accum_every = 8,         # effective batch size of 32
    data_max_length = 320 * 32,
    save_results_every = 2,
    save_model_every = 4,
    num_train_steps = 1000
).cuda()
# NOTE: I changed num_train_steps to 9 (aka 8 + 1) from 10000 to make things go faster for demo purposes
# adjusting save_*_every variables for the same reason

trainer.train()

