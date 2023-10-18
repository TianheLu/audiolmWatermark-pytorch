import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from audiolm_pytorch import SoundStream, SoundStreamTrainer
import torch


soundstream = SoundStream(
    codebook_size = 1024,
    rq_num_quantizers = 8,
    rq_groups = 2,                # this paper proposes using multi-headed residual vector quantization - https://arxiv.org/abs/2305.02765
    attn_window_size = 128,       # local attention receptive field at bottleneck
    attn_depth = 2                # 2 local attention transformer blocks - the soundstream folks were not experts with attention, so i took the liberty to add some. encodec went with lstms, but attention should be better
)

sum = 0
for name, param in soundstream.named_parameters():
    num = 1
    for size in param.shape:
        num *= size
    sum += num
    print("{:30s}:{}".format(name, param.shape))
print("total param num {}".format(sum))

trainer = SoundStreamTrainer(
    soundstream,
    folder = 'train-clean-100',
    batch_size = 4,
    grad_accum_every = 8,         # effective batch size of 32
    # data_max_length_seconds = 2,  # train on 2 second audio
    data_max_length= 320 * 32,
    save_results_every = 50,
    save_model_every = 100,
    num_train_steps = 1_000_000
).cuda()


trainer.train()



audio = torch.randn(10080).cuda()
recons = soundstream(audio, return_recons_only = True) # (1, 10080) - 1 channel

