from utils.classifier_utils import (
    create_classifier,
    create_classifier_dataloaders,
    compute_val_accuracy,
    perform_train_step,
    create_generated_batch,
    create_real_batch,
)
from generator import Generator
from utils.parser import get_omniglot_classifier_args
import torch.nn as nn
import torch
import os
import torch.optim as optim
import numpy as np
import scipy.stats
import time


# To maintain reproducibility
torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load input args
args = get_omniglot_classifier_args()

generator_path = args.generator_path
dataset_path = args.dataset_path
data_start_index = args.data_start_index
num_classes = args.num_training_classes
generated_batches_per_real = args.generated_batches_per_real
num_epochs = args.epochs
num_val = args.val_samples_per_class
num_train = args.train_samples_per_class
num_bootstrap_samples = args.num_bootstrap_samples
progress_frequency = args.progress_frequency
batch_size = args.batch_size


raw_data = np.load(dataset_path)
raw_data = raw_data[data_start_index:]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dagan_generator = torch.load(generator_path, map_location=device)
dagan_generator.eval()

loss_function = nn.CrossEntropyLoss()

start_time = int(time.time())
last_progress_time = start_time


val_accuracies_list = []

# Run routine both with and without augmentation
for real_batch_rate in (1, generated_batches_per_real + 1):
    print(
        "Training %d classifiers with %d generated batches per real"
        % (num_bootstrap_samples, real_batch_rate - 1)
    )
    val_accuracies = []

    # Train {num_bootstrap_samples} classifiers each with different samples of data
    for bootstrap_sample in range(num_bootstrap_samples):
        print("\nBootstrap #%d" % (bootstrap_sample + 1))
        classifier = create_classifier(num_classes).to(device)
        train_dataloader, val_dataloader = create_classifier_dataloaders(
            raw_data, num_classes, num_train, num_val, batch_size
        )
        optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.99))

        # Train on the dataset {num_epochs} times
        for epoch in range(num_epochs):
            training_loss = perform_train_step(
                classifier,
                train_dataloader,
                optimizer,
                loss_function,
                device,
                real_batch_rate,
                dagan_generator,
            )

            if epoch % progress_frequency == 0:
                print("[%d] train loss: %.5f" % (epoch + 1, training_loss))
                compute_val_accuracy(val_dataloader, classifier, device, loss_function)

                last_progress_time = int(time.time())
                print(
                    f"Elapsed time: {(last_progress_time - start_time) / 60:.2f} minutes\n"
                )

        print("[%d] train loss: %.5f" % (num_epochs, training_loss))
        val_accuracies.append(
            compute_val_accuracy(val_dataloader, classifier, device, loss_function)
        )

        # Remove current net from gpu memory
        del classifier
    val_accuracies_list.append(val_accuracies)

# Summarize and print results
pvalue = scipy.stats.ttest_ind(
    val_accuracies_list[0], val_accuracies_list[1], equal_var=False
).pvalue
print(
    "Trained %d classifiers with and without augmentation using %d samples per class"
    % (num_bootstrap_samples, num_train)
)
print("Average accuracy without augmentation: %.5f" % (np.mean(val_accuracies_list[0])))
print("Average accuracy with augmentation: %.5f" % (np.mean(val_accuracies_list[1])))
print(
    "Confidence level that augmentation has higher accuracy, using 2 sample t-test: %.3f%%"
    % (100 * (1 - pvalue))
)
