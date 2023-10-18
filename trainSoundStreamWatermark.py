import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
from audiolm_pytorch import SoundStreamWatermark, SoundStreamWatermarkTrainer
import torch


soundStreamWatermark = SoundStreamWatermark(
    msg_len = 4,
    msg_loss_weight = 1000,
    recon_loss_weight = 10,
    feature_loss_weight = 50,
    codebook_size = 1024,
    rq_num_quantizers = 8,
    rq_groups = 2,                # this paper proposes using multi-headed residual vector quantization - https://arxiv.org/abs/2305.02765
    attn_window_size = 128,       # local attention receptive field at bottleneck
    attn_depth = 2                # 2 local attention transformer blocks - the soundstream folks were not experts with attention, so i took the liberty to add some. encodec went with lstms, but attention should be better
)

sum = 0
for name, param in soundStreamWatermark.named_parameters():
    num = 1
    for size in param.shape:
        num *= size
    sum += num
    print("{:30s}:{}".format(name, param.shape))
print("total param num {}".format(sum))

trainer = SoundStreamWatermarkTrainer(
    soundStreamWatermark,
    folder = 'train-clean-100',
    batch_size = 8,
    grad_accum_every = 8,         # effective batch size of 32
    # data_max_length_seconds = 2,  # train on 2 second audio
    data_max_length= 320 * 32 * 2,
    save_results_every = 50,
    save_model_every = 100,
    num_train_steps = 1_000_000,
    evaluation_nums = 5  # evaluate more audios, to calculate msg acc more accurately
).cuda()


trainer.train()



audio = torch.randn(10080).cuda()
recons = soundStreamWatermark(audio, return_recons_only = True) # (1, 10080) - 1 channel

