from torchvision.models.densenet import DenseNet
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import warnings
import torch


class ModifiedDenseNet(DenseNet):
    """
    Densenet architecture modified to acccept a flag for whether
    a given input image is a real or generated sample.

    Extra dense layers are also added at the end to allow the input flag
    to interact with the extracted image features before outputting a prediction.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(self.classifier.in_features + 1, 20)),
                    ("norm1", nn.BatchNorm1d(20)),
                    ("linear2", nn.Linear(20, self.classifier.out_features)),
                ]
            )
        )

    def forward(self, x, flags):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = torch.cat((out, flags.float()), dim=1)
        out = self.classifier(out)
        return out


class AddGaussianNoiseTransform:
    """
    Adds random gaussian noise to an image.

    Useful for data augmentation during the training phase.
    """

    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, arr):
        return (
            np.array(arr)
            + np.random.randn(*arr.shape).astype("float32") * self.std
            + self.mean
        )

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class ClassifierDataset(Dataset):
    def __init__(self, examples, labels, transform=None):
        self.examples = examples
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        sample = self.examples[idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.transform:
                sample = self.transform(sample)

        return sample, self.labels[idx]


pre_train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ]
)

train_transform = transforms.Compose(
    [
        AddGaussianNoiseTransform(0, 0.1),
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ]
)


def gen_out_to_numpy(tensor):
    """
    Convert tensor of images to numpy format.
    """
    return ((tensor * 0.5) + 0.5).squeeze(1).unsqueeze(-1).numpy()


def create_real_batch(samples):
    """
    Given a batch of images, apply the train transform.
    """
    np_samples = gen_out_to_numpy(samples)
    new_samples = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(np_samples.shape[0]):
            new_samples.append(train_transform(np_samples[i]))
    return torch.stack(new_samples)


def create_generated_batch(samples, generator, device="cpu"):
    """
    Given a batch of images, run them through a dagan
    and apply the train transform.
    """
    z = torch.randn((samples.shape[0], generator.z_dim))
    with torch.no_grad():
        g_out = generator(samples.to(device), z.to(device)).cpu()
    np_out = gen_out_to_numpy(g_out)

    new_samples = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(np_out.shape[0]):
            new_samples.append(train_transform(np_out[i]))
    return torch.stack(new_samples)


def perform_train_step(
    net, train_dataloader, optimizer, loss_function, device, real_batch_rate, g
):
    """
    Perform one epoch of training using the full dataset.
    """
    running_loss = 0.0
    net.train()
    for i, data in enumerate(train_dataloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0], data[1].to(device)
        if i % real_batch_rate == 0:
            inputs = create_real_batch(inputs).to(device)
            flags = torch.ones((inputs.shape[0], 1)).to(device)
        else:
            inputs = create_generated_batch(inputs, g, device).to(device)
            flags = torch.zeros((inputs.shape[0], 1)).to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs, flags)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    return running_loss / len(train_dataloader.dataset)


def compute_val_accuracy(val_dataloader, net, device, loss_function):
    """
    Compute accuracy on the given validation set.
    """
    val_loss = 0.0
    success = 0
    success_topk = 0
    k = 5
    net.eval()
    val_dataset = val_dataloader.dataset
    for i, data in enumerate(val_dataloader):
        with torch.no_grad():
            inputs, labels = data[0].to(device), data[1].to(device)
            flags = torch.ones((inputs.shape[0], 1)).to(device)
            outputs = net(inputs, flags)
            _, predicted = torch.max(outputs, 1)
            _, predicted_topk = torch.topk(outputs, axis=1, k=k)
            success += sum(
                [int(predicted[i] == labels[i]) for i in range(len(predicted))]
            )
            success_topk += sum(
                [
                    int(labels[i] in predicted_topk[i])
                    for i in range(len(predicted_topk))
                ]
            )
            loss = loss_function(outputs, labels)
            val_loss += loss.item()
    print("val loss: %.5f" % (val_loss / len(val_dataset)))
    print("val acc: %.5f" % (success / len(val_dataset)))
    print("val acc top%d: %.5f" % (k, success_topk / len(val_dataset)))
    return success / len(val_dataset)


def create_classifier(num_classes):
    """
    Create a modified densenet with the given number of classes.
    """
    classifier = ModifiedDenseNet(
        growth_rate=16,
        block_config=(3, 3, 3, 3),
        num_classes=num_classes,
        drop_rate=0.3,
    )
    classifier.features[0] = nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    return classifier


def create_classifier_dataloaders(
    raw_data, num_classes, num_train, num_val, batch_size
):
    """
    Create train and validation dataloaders with the given raw data.
    """
    train_X = []
    train_y = []
    val_X = []
    val_y = []

    for i in range(num_classes):
        # Shuffle data so different examples are chosen each time
        class_data = list(raw_data[i])
        np.random.shuffle(class_data)

        train_X.extend(class_data[:num_train])
        train_y.extend([i] * num_train)
        val_X.extend(class_data[-num_val:])
        val_y.extend([i] * num_val)

    train_dataloader = DataLoader(
        ClassifierDataset(train_X, train_y, pre_train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )
    val_dataloader = DataLoader(
        ClassifierDataset(val_X, val_y, val_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
    )

    return train_dataloader, val_dataloader
