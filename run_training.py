from trainer import Trainer
from discriminator import create_d
from generator import Generator
from dataset import OmniglotDataset, create_dataloader
import torchvision.transforms as transforms
import torch
import torch.optim as optim
import numpy as np


# To maintain reproducibility
torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

raw_data = np.load("datasets/omniglot_data.npy")

in_channels = raw_data.shape[-1]
dim = raw_data.shape[1]
img_size = 64
num_classes = 1200
batch_size = 32
epochs = 200

g = Generator(dim=img_size, channels=in_channels)
d = create_d(in_channels=in_channels)

train_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(img_size),
        transforms.RandomAffine([-10, 10], translate=[0.2, 0.2]),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ]
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_dataloader = create_dataloader(raw_data, num_classes, train_transform, batch_size)

g_opt = optim.Adam(g.parameters(), lr=0.0001, betas=(0.0, 0.9))
d_opt = optim.Adam(d.parameters(), lr=0.0001, betas=(0.0, 0.9))

trainer = Trainer(g, d, g_opt, d_opt, batch_size, device)
trainer.train(train_dataloader, epochs=200)
