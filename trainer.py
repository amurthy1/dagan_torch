import imageio
import numpy as np
import torch
import time
import torch.nn as nn
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.autograd import grad as torch_grad


class Trainer:
    def __init__(
        self,
        generator,
        discriminator,
        gen_optimizer,
        dis_optimizer,
        batch_size,
        device="cpu",
        gp_weight=10,
        critic_iterations=5,
        print_every=50,
        val_images=None,
    ):
        self.device = device
        self.g = generator.to(device)
        self.g_opt = gen_optimizer
        self.d = discriminator.to(device)
        self.d_opt = dis_optimizer
        self.batch_size = batch_size
        self.losses = {"G": [], "D": [], "GP": [], "gradient_norm": []}
        self.num_steps = 0
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every
        self.val_images = val_images

    def _critic_train_iteration(self, x1, x2):
        """ """
        # Get generated data
        generated_data = self.sample_generator(x1)

        # Calculate probabilities on real and generated data
        # data = Variable(data)
        # if self.use_cuda:
        #     data = data.cuda()
        d_real = self.d(x1, x2)
        d_generated = self.d(x1, generated_data)

        # Get gradient penalty
        gradient_penalty = self._gradient_penalty(x1, x2, generated_data)
        self.losses["GP"].append(gradient_penalty.item())

        # Create total loss and optimize
        self.d_opt.zero_grad()
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
        d_loss.backward()

        self.d_opt.step()

        # Record loss
        self.losses["D"].append(d_loss.item())

    def _generator_train_iteration(self, x1):
        """ """
        self.g_opt.zero_grad()

        # Get generated data
        generated_data = self.sample_generator(x1)

        # Calculate loss and optimize
        d_generated = self.d(x1, generated_data)
        g_loss = -d_generated.mean()
        print(g_loss)
        g_loss.backward()
        self.g_opt.step()

        # Record loss
        self.losses["G"].append(g_loss.item())

    def _gradient_penalty(self, x1, x2, generated_data):
        # Calculate interpolation
        alpha = torch.rand(self.batch_size, 1, 1, 1)
        alpha = alpha.expand_as(x2).to(self.device)
        # if self.use_cuda:
        #     alpha = alpha.cuda()
        interpolated = alpha * x2.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True).to(self.device)
        # if self.use_cuda:
        #     interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = self.d(x1, interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(self.batch_size, -1)
        self.losses["gradient_norm"].append(gradients.norm(2, dim=1).mean().item())

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def _train_epoch(self, data_loader):
        for i, data in enumerate(data_loader):
            self.num_steps += 1
            x1, x2 = data[0].to(self.device), data[1].to(self.device)
            self._critic_train_iteration(x1, x2)
            # Only update generator every |critic_iterations| iterations
            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(x1)

            if i % self.print_every == 0:
                print("Iteration {}".format(i + 1))
                print("D: {}".format(self.losses["D"][-1]))
                print("GP: {}".format(self.losses["GP"][-1]))
                print("Gradient norm: {}".format(self.losses["gradient_norm"][-1]))
                if self.num_steps > self.critic_iterations:
                    print("G: {}".format(self.losses["G"][-1]))

    def train(self, data_loader, epochs, save_training_gif=True):
        # if save_training_gif:
        #     # Fix latents to see how image generation improves during training
        #     fixed_latents = Variable(self.g.sample_latent(64))
        #     if self.use_cuda:
        #         fixed_latents = fixed_latents.cuda()
        #     training_progress_images = []

        start_time = int(time.time())
        for epoch in range(1, epochs + 1):
            # self.g.eval()
            # with torch.no_grad():
            #     self.display_generations(data_loader, num_generations=4)
            # self.g.train()
            print("\nEpoch {}".format(epoch))
            print(f"Elapsed time: {(time.time() - start_time) / 60:.2f} minutes\n")
            self._train_epoch(data_loader)
            self.save_checkpoints(epoch)

            # if save_training_gif:
            #
            # Generate batch of images and convert to grid
            #     img_grid = make_grid(self.g(fixed_latents).cpu().data)
            #     # Convert to numpy and transpose axes to fit imageio convention
            #     # i.e. (width, height, channels)
            #     img_grid = np.transpose(img_grid.numpy(), (1, 2, 0))
            #     # Add image grid to training progress
            #     training_progress_images.append(img_grid)

        # if save_training_gif:
        #     imageio.mimsave('./training_{}_epochs.gif'.format(epochs),
        #                     training_progress_images)

    def sample_generator(self, input_images):
        # input_images.to(self.device)
        z = torch.randn((self.batch_size, self.g.z_dim)).to(self.device)
        # latent_samples = Variable(self.g.sample_latent(num_samples))
        # if self.use_cuda:
        #     latent_samples = latent_samples.cuda()
        return self.g(input_images, z)

    def save_img(self, arr, filename):
        arr = (arr * 0.5) + 0.5
        arr = np.uint8(arr * 255)
        Image.fromarray(arr, mode="L").save(filename)

    def display_generations(self, data_loader, num_generations):
        train_idx = torch.randint(0, len(data_loader.dataset), (1,))[0]
        train_img = display_transform(data_loader.dataset.x1_examples[train_idx])
        self.save_img(train_img[0])

        z = torch.randn((1, self.g.z_dim)).to(self.device)
        inp = train_img.unsqueeze(0).to(self.device)
        train_gen = self.g(inp, z).cpu()[0]
        self.render_img(train_gen[0])

    def save_checkpoints(self, epoch):
        torch.save(self.g, f"checkpoints/g_{epoch}.pt")
        torch.save(self.d, f"checkpoints/d_{epoch}.pt")
