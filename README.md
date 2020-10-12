# Data Augmentation GAN in PyTorch

<img src="resources/dagan_tracking_images.png" width=560 height=56/>
<img src="resources/dagan_training_progress.gif" width=560 height=56/>

<i>Time-lapse of DAGAN generations on the omniglot dataset over the course of the training process.</i>


## Table of Contents
1. [Intro](#intro)
2. [Background](#background)

## 1. Intro <a name="intro"></a>

This is a PyTorch implementation of Data Augmentation GAN (DAGAN), which was first proposed in [this paper](https://arxiv.org/abs/1711.04340) with a [corresponding tensorflow implementation](https://github.com/AntreasAntoniou/DAGAN).

This repo uses the same generator and discriminator architecture of the original tf implementation, while also including a classifier script for the omniglot dataset to test out the quality of a trained DAGAN.

## 2. Background <a name="background"></a>

The motivation for this work is to train a [Generative Adversarial Network (GAN)](https://en.wikipedia.org/wiki/Generative_adversarial_network) which takes in an image of a given class (e.g. a specific letter in an alphabet) and outputs another image of the same class that is sufficiently different looking than the input. This GAN is then used as a tool for data augmentation when training an image classifier.

Standard data augmentation includes methods such as adding noise to, rotating, or cropping images, which increases variation in the training samples and improves the robustness of the trained classifier. Randomly passing some images through the DAGAN generator before using them in training serves a similar purpose.
