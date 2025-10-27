# VQ-VAE A Posteriori with Geodesic Quantization

Deep Learning and Applied AI project - Sapienza.

This project revisits the VQ-VAE pipeline by introducing a **post-hoc quantization approach** that leverages geodesic distances in the learned latent manifold. Then compare reconstruction quality, sample fidelity, and codebook utilization against standard end-to-end **VQ-VAE** on MNIST and CelebA datasets.

Checkpoints and other .pt files could not be included due to github file size limit (100 mb).

UV and wandb were very useful tools in this project.

## Visual Results

### MNIST

#### Reconstruction Comparison
![MNIST Reconstructions](data/recons_mnist/reconstructions_comparison.png)
*Top to bottom: Original, VAE, VAE+Geodesic, VQ-VAE*

#### Geodesic Quantization - PixelCNN Sampling
![MNIST Geodesic Sampling](data/samples_mnist/geodesic_pixelcnn_vs_random.png)
*PixelCNN samples (top) vs random codes (bottom) - Geodesic quantization*

#### Geodesic PixelCNN Samples
<img src="data/samples_mnist/geodesic_pixelcnn_samples_grid_t1.0.png" width="400">

*Geodesic PixelCNN samples*

#### Geodesic Quantization - Temperature Comparison
![MNIST Temperature Comparison](data/samples_mnist/geodesic_temperature_comparison.png)
*Effect of temperature on sample diversity*

#### VQ-VAE - PixelCNN Sampling
![MNIST VQ-VAE Sampling](data/samples_mnist/vqvae_pixelcnn_vs_random.png)
*PixelCNN samples (top) vs random codes (bottom) - VQ-VAE*

#### Latent Space Interpolations
![MNIST Interpolations](data/samples_mnist/interpolation_data.png)
*Top to bottom: Original images, VAE+Geodesic (quantized latents), VQ-VAE. Each row shows smooth interpolation between two random samples.*

---

### CelebA

#### Reconstruction Comparison
![CelebA Reconstructions](data/recons_celeba/reconstructions_comparison_celeba.png)
*Top to bottom: Original, VAE, VAE+Geodesic, VQ-VAE*

#### Geodesic Quantization - PixelCNN Sampling
![CelebA Geodesic Sampling](data/samples_celeba/geodesic_pixelcnn_vs_random_celeba.png)
*PixelCNN samples (top) vs random codes (bottom) - Geodesic quantization*

#### Geodesic PixelCNN Samples
<img src="data/samples_celeba/geodesic_pixelcnn_samples_celeba_grid_t1.0.png" width="400">

*Geodesic PixelCNN samples*

#### Geodesic Quantization - Temperature Comparison
![CelebA Temperature Comparison](data/samples_celeba/geodesic_temperature_comparison_celeba.png)
*Effect of temperature on sample diversity*

#### VQ-VAE - PixelCNN Sampling
![CelebA VQ-VAE Sampling](data/samples_celeba/vqvae_pixelcnn_vs_random_celeba.png)
*PixelCNN samples (top) vs random codes (bottom) - VQ-VAE*

#### Latent Space Interpolations
![CelebA Interpolations](data/samples_celeba/interpolation_data_celeba.png)
*Top to bottom: Original images, VAE+Geodesic (quantized latents), VQ-VAE. Each row shows smooth interpolation between two random face samples.*

---

### CIFAR-10 with beta vae (b=0.1) 
Just for testing... not included in the report because of blurry results

#### Reconstruction Comparison
![CIFAR-10 Reconstructions](data/recons_cifar10/reconstructions_comparison_cifar10.png)
*Top to bottom: Original, VAE, VAE+Geodesic, VQ-VAE*

#### Geodesic Quantization - PixelCNN Sampling
![CIFAR-10 Geodesic Sampling](data/samples_cifar10/geodesic_pixelcnn_vs_random_cifar10.png)
*PixelCNN samples (top) vs random codes (bottom) - Geodesic quantization*

#### Geodesic PixelCNN Samples
<img src="data/samples_cifar10/geodesic_pixelcnn_samples_cifar10_grid_t1.0.png" width="400">

*Geodesic PixelCNN samples*

#### Geodesic Quantization - Temperature Comparison
![CIFAR-10 Temperature Comparison](data/samples_cifar10/geodesic_temperature_comparison_cifar10.png)
*Effect of temperature on sample diversity*

#### VQ-VAE - PixelCNN Sampling
![CIFAR-10 VQ-VAE Sampling](data/samples_cifar10/vqvae_pixelcnn_vs_random_cifar10.png)
*PixelCNN samples (top) vs random codes (bottom) - VQ-VAE*

#### Latent Space Interpolations
![CIFAR-10 Interpolations](data/samples_cifar10/interpolation_data_cifar10.png)
*Top to bottom: Original images, VAE+Geodesic (quantized latents), VQ-VAE. Each row shows smooth interpolation between two random samples.*


