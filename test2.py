# imports
import math
import wave
import struct
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import urllib.request
import tarfile
from audiolm_pytorch import SoundStream, SoundStreamTrainer, HubertWithKmeans, SemanticTransformer, SemanticTransformerTrainer, HubertWithKmeans, CoarseTransformer, CoarseTransformerWrapper, CoarseTransformerTrainer, FineTransformer, FineTransformerWrapper, FineTransformerTrainer, AudioLM
from torch import nn
import torch
import torchaudio


# define all dataset paths, checkpoints, etc
dataset_folder = "placeholder_dataset"
soundstream_ckpt = "results/soundstream.8.pt" # this can change depending on number of steps
hubert_ckpt = 'hubert/hubert_base_ls960.pt'
hubert_quantizer = f'hubert/hubert_base_ls960_L9_km500.bin' # listed in row "HuBERT Base (~95M params)", column Quantizer


# hubert checkpoints can be downloaded at
# https://github.com/facebookresearch/fairseq/tree/main/examples/hubert
if not os.path.isdir("hubert"):
    os.makedirs("hubert")
if not os.path.isfile(hubert_ckpt):
    hubert_ckpt_download = f"https://dl.fbaipublicfiles.com/{hubert_ckpt}"
    urllib.request.urlretrieve(hubert_ckpt_download, f"./{hubert_ckpt}")
if not os.path.isfile(hubert_quantizer):
    hubert_quantizer_download = f"https://dl.fbaipublicfiles.com/{hubert_quantizer}"
    urllib.request.urlretrieve(hubert_quantizer_download, f"./{hubert_quantizer}")

wav2vec = HubertWithKmeans(
    checkpoint_path = f'./{hubert_ckpt}',
    kmeans_path = f'./{hubert_quantizer}'
)

semantic_transformer = SemanticTransformer(
    num_semantic_tokens = wav2vec.codebook_size,
    dim = 1024,
    depth = 6
).cuda()


trainer = SemanticTransformerTrainer(
    transformer = semantic_transformer,
    wav2vec = wav2vec,
    folder = dataset_folder,
    batch_size = 1,
    data_max_length = 320 * 32,
    num_train_steps = 1
)

trainer.train()

