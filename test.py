import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = 32,
    timesteps = 1000,    # number of steps
    objective = 'score_matching',
)

path_to_img = '/Users/macheng/CIFAR-10-images-master/train/'

trainer = Trainer(
    diffusion,
    path_to_img,
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = True              # whether to calculate fid during training
)

trainer.train()

sampled_images = diffusion.sample(batch_size = 4)
sampled_images.shape # (4, 3, 128, 128)


# loss = diffusion(training_images)
# loss.backward()
# after a lot of training

# def unpickle(file):
#     import pickle
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict

# # training_images = torch.rand(8, 3, 128, 128) # images are normalized from 0 to 1
# training_images = unpickle(path_to_img)[b'data']
# num_of_imgs = training_images.shape[0]
# training_images = training_images.reshape(num_of_imgs, 3, 1024)
# training_images = training_images.reshape(num_of_imgs, 3, 32, 32) / 255
# training_images = torch.from_numpy(training_images).float()