import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from audiolm_pytorch import SoundStream, SoundStreamTrainer
import torch
import torchaudio
import torch.nn.functional as F
from torchaudio.functional import resample
from audiolm_pytorch.utils import curtail_to_multiple
from einops import rearrange, reduce
path_of_model = "results/soundstream.44000.pt"
path_of_input = "dev-clean/LibriSpeech/dev-clean/84/121123/84-121123-0001.flac"
path_of_input = "train-clean-100/LibriSpeech/train-clean-100/5561/41616/5561-41616-0021.flac"
name_of_output = path_of_input.split("/")[-1]
path_of_output = "dev-results/" + name_of_output
soundstream = SoundStream.init_and_load_from(path_of_model)


def exists(val):
    return val is not None
def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)


def generate_audio_tensor(path_of_input, target_sample_hz = 16000, max_target_sample_hz = 16000):

    data, sample_hz = torchaudio.load(path_of_input)
    if data.shape[0] > 1:
        # the audio has more than 1 channel, convert to mono
        data = reduce(data, 'c ... -> 1 ...', 'mean')
    # first resample data to the max target freq

    data = resample(data, sample_hz, max_target_sample_hz)
    sample_hz = max_target_sample_hz
    max_length = 320 * 32 * 20
    audio_length = data.size(1)
    if exists(max_length):
        if audio_length > max_length:
            max_start = audio_length - max_length
            # start = torch.randint(0, max_start, (1,))
            start = torch.tensor([0])
            data = data[:, start:start + max_length]
        else:
            data = F.pad(data, (0, max_length - audio_length), 'constant')

    data = rearrange(data, '1 ... -> ...')


    num_outputs = 1
    data = cast_tuple(data, num_outputs)
    target_sample_hz = cast_tuple(target_sample_hz)
    data_tuple = tuple(resample(d, sample_hz, target_sample_hz) for d, target_sample_hz in zip(data, target_sample_hz))

    output = []
    seq_len_multiple_of = 320
    seq_len_multiple_of = cast_tuple(seq_len_multiple_of, num_outputs)

    for data, seq_len_multiple_of in zip(data_tuple, seq_len_multiple_of):
        if exists(seq_len_multiple_of):
            data = curtail_to_multiple(data, seq_len_multiple_of)

        output.append(data.float())

    output = tuple(output)

    if num_outputs == 1:
        audio = output[0]
    else:
        audio = output
    return audio


if __name__ == "__main__":
    audio = generate_audio_tensor(path_of_input)
    generated_wav = soundstream(audio, return_recons_only = True) # (1, 10080) - 1 channel
    print(audio)
    print(generated_wav)

    sample_rate = 16000
    print(os.path.abspath(path_of_output))
    torchaudio.save(path_of_output, generated_wav.cpu(), sample_rate)