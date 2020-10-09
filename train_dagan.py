from dagan_trainer import DaganTrainer
from discriminator import Discriminator
from generator import Generator
from dataset import create_dagan_dataloader
from utils.parser import get_dagan_args
import torchvision.transforms as transforms
import torch
import os
import torch.optim as optim
import numpy as np


# To maintain reproducibility
torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load input args
args = get_dagan_args()

dataset_path = args.dataset_path
raw_data = np.load(dataset_path).copy()

final_generator_path = args.final_model_path
save_checkpoint_path = args.save_checkpoint_path
load_checkpoint_path = args.load_checkpoint_path
in_channels = raw_data.shape[-1]
img_size = args.img_size or raw_data.shape[2]
num_training_classes = args.num_training_classes
num_val_classes = args.num_val_classes
batch_size = args.batch_size
epochs = args.epochs
dropout_rate = args.dropout_rate
max_pixel_value = args.max_pixel_value
should_display_generations = not args.suppress_generations

# Input sanity checks
final_generator_dir = os.path.dirname(final_generator_path) or os.getcwd()
if not os.access(final_generator_dir, os.W_OK):
    raise ValueError(final_generator_path + " is not a valid filepath.")

if num_training_classes + num_val_classes > raw_data.shape[0]:
    raise ValueError(
        "Expected at least %d classes but only had %d."
        % (num_training_classes + num_val_classes, raw_data.shape[0])
    )


g = Generator(dim=img_size, channels=in_channels, dropout_rate=dropout_rate)
d = Discriminator(dim=img_size, channels=in_channels * 2, dropout_rate=dropout_rate)

mid_pixel_value = max_pixel_value / 2
train_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            (mid_pixel_value,) * in_channels, (mid_pixel_value,) * in_channels
        ),
    ]
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_dataloader = create_dagan_dataloader(
    raw_data, num_training_classes, train_transform, batch_size
)

g_opt = optim.Adam(g.parameters(), lr=0.0001, betas=(0.0, 0.9))
d_opt = optim.Adam(d.parameters(), lr=0.0001, betas=(0.0, 0.9))

val_data = raw_data[num_training_classes : num_training_classes + num_val_classes]
flat_val_data = val_data.reshape(
    (val_data.shape[0] * val_data.shape[1], *val_data.shape[2:])
)

display_transform = train_transform

trainer = DaganTrainer(
    generator=g,
    discriminator=d,
    gen_optimizer=g_opt,
    dis_optimizer=d_opt,
    batch_size=batch_size,
    device=device,
    critic_iterations=5,
    print_every=75,
    num_tracking_images=10,
    save_checkpoint_path=save_checkpoint_path,
    load_checkpoint_path=load_checkpoint_path,
    display_transform=display_transform,
    should_display_generations=should_display_generations,
)
trainer.train(data_loader=train_dataloader, epochs=epochs, val_images=flat_val_data)

# Save final generator model
torch.save(trainer.g, final_generator_path)
