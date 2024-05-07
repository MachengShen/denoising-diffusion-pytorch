import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer, RepresentationEncoder

def save_images(imgs):
    import torchvision
    from PIL import Image
    imgs = imgs.cpu().float()

    # 遍历 sampled_images 中的每个图像
    for i, single_image in enumerate(imgs):
        img_np = single_image.permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype('uint8')
        img = Image.fromarray(img_np)
        img.save(f'sampled_image_{i}.png')

    print('Images saved successfully.')

image_size = 32
objective = 'representation_learning'
latent_code_dim = 128 if objective == 'representation_learning' else 0
model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = False,
    latent_code_dim=latent_code_dim,
)

if objective == 'representation_learning':
    representation_encoder = RepresentationEncoder(image_size = image_size,
                                                   latent_dim = latent_code_dim,)
else: 
    representation_encoder = None

diffusion = GaussianDiffusion(
    model,
    image_size = image_size,
    timesteps = 1000,    # number of steps
    objective = objective,
    representation_encoder=representation_encoder
)

training_images = torch.rand(8, 3, image_size, image_size) # images are normalized from 0 to 1
loss = diffusion(training_images)

path_to_img = '/Users/macheng/CIFAR-10-images-master/train/'
# path_to_img = '/root/CIFAR-10-images-master/train/'

trainer = Trainer(
    diffusion,
    path_to_img,
    train_batch_size = 64,
    train_lr = 10e-5,
    train_num_steps = 100000,         # total training steps 7e5
    gradient_accumulate_every = 2,    # gradient accumulation steps
    save_and_sample_every = 10000,
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = False              # whether to calculate fid during training
)

trainer.train()

print("-----------------training complete---------------")
sampled_images = diffusion.sample(batch_size = 4)
print(sampled_images.shape) # (4, 3, 128, 128)

save_images(sampled_images)



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