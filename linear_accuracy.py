from pathlib import Path
from functools import partial
from multiprocessing import cpu_count

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms as T

from PIL import Image

from denoising_diffusion_pytorch.denoising_diffusion_pytorch import convert_image_to_fn, exists

class LinearModel(nn.Module):
    def __init__(self, feature_dim = 128):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(feature_dim, 10)  # 3072 for flattened 32x32x3 images, 10 for CIFAR-10 classes

    def forward(self, x):
        x = self.linear(x)
        return x
        
class LabeledCifarDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])
        self.label_map = {"airplane": 0,
                          "automobile": 1,
                          "bird": 2,
                          "cat": 3,
                          "deer": 4,
                          "dog": 5,
                          "frog": 6,
                          "horse": 7,
                          "ship": 8,
                          "truck": 9,}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        label = self.label_map[str(path).split("/")[-2]]
        return self.transform(img), label


if __name__ == '__main__':
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # path_to_img = '/Users/macheng/CIFAR-10-images-master/'
    path_to_img = '/root/CIFAR-10-images-master/'

    folder = path_to_img + 'train/'
    test_folder = path_to_img + 'test/'
    image_size = 32
    convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(3)
    ds = LabeledCifarDataset(folder, image_size, augment_horizontal_flip = False, convert_image_to = convert_image_to)
    dl = DataLoader(ds, batch_size = 32, shuffle = True, pin_memory = True, num_workers = cpu_count())
    test_ds = ds = LabeledCifarDataset(test_folder, image_size, augment_horizontal_flip = False, convert_image_to = convert_image_to)
    test_dl = DataLoader(ds, batch_size = 32, shuffle = False, pin_memory = True, num_workers = cpu_count())
    # Create an instance of the model
    model = LinearModel(feature_dim = 128).to(device)
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    encoder = torch.load('representation_encoder_100000.pth')
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dl:
            optimizer.zero_grad()
            outputs = model(encoder(inputs.to(device)))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(dl):.4f}')
        
        # Evaluation
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_dl:
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to the device
                outputs = model(encoder(inputs))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy: {100 * correct / total:.2f}%')