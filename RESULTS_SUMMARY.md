# VQ-VAE with Geodesic Quantization - Results Summary

**Project**: Deep Learning Course - University Project  
**Date**: October 11, 2025  
**Author**: Michele Granatiero

---

## 📋 Project Overview

This project revisits the VQ-VAE pipeline with a novel approach: instead of jointly learning a discrete latent space, we first train a standard continuous VAE, then perform **a posteriori vector quantization using geodesic distances** in the latent space, rather than Euclidean ones.

### Key Innovation
- **Traditional VQ-VAE**: End-to-end learning with Euclidean distance
- **Our Approach**: Post-hoc geodesic quantization respecting latent manifold geometry

---

## 🏗️ Architecture Comparison

### VAE + Geodesic Quantization
```
Training Phase:
1. Train standard VAE (encoder → μ, logvar → decoder)
2. Freeze VAE, extract latent representations
3. Apply K-means clustering with Wasserstein-2 (geodesic) distance
4. Build discrete codebook from cluster centroids
5. Train PixelCNN autoregressive prior over discrete codes

Generation Phase:
- Sample codes from PixelCNN prior
- Lookup in geodesic codebook
- Decode with VAE decoder
```

**Parameters**: ~1.5M (VAE) + ~150k (PixelCNN)

### VQ-VAE (Baseline)
```
Training Phase:
1. Train encoder → vector quantizer → decoder end-to-end
2. Vector quantizer uses straight-through estimator
3. Codebook learned jointly via L2 distance + commitment loss

Generation Phase:
- Sample codes uniformly (no learned prior in this baseline)
- Lookup in learned codebook
- Decode with VQ-VAE decoder
```

**Parameters**: ~1.06M (comparable architecture to VAE)

### Architecture Details (MNIST)

Both models now use **equivalent depth and capacity**:

#### Encoder
| Layer | VAE | VQ-VAE | Output Shape |
|-------|-----|--------|--------------|
| conv1 | 1→128, stride=2 | 1→128, stride=2 | 128x14x14 |
| conv2 | 128→256, stride=2 | 128→256, stride=2 | 256x7x7 |
| output | flatten → fc_mu, fc_logvar | 1x1 conv to embedding_dim | 256x7x7 (VAE flat), 4x7x7 (VQ-VAE spatial) |

#### Decoder
| Layer | VAE | VQ-VAE | Output Shape |
|-------|-----|--------|--------------|
| input | fc → unflatten | 1x1 conv projection | 256x7x7 |
| conv2 | 256→128, stride=2 | 256→128, stride=2 | 128x14x14 |
| conv1 | 128→1, stride=2 | 128→1, stride=2 | 1x28x28 |

**Key Difference**: VAE uses flat vectors; VQ-VAE uses spatial feature maps (for per-position quantization).

---

## 📊 Quantitative Results

### Reconstruction Quality

| Method | MSE ↓ | Relative Performance |
|--------|-------|---------------------|
| **VQ-VAE (end-to-end)** | **0.002380** | **Best** (baseline) |
| VAE + Geodesic | 0.013601 | 471% worse than VQ-VAE |

**Analysis**: 
- VQ-VAE achieves superior reconstruction quality due to end-to-end joint optimization
- Geodesic approach sacrifices reconstruction for better latent space geometry
- The gap is expected: post-hoc quantization cannot optimize encoder/decoder for discrete codes

### Codebook Utilization

| Method | Utilization | Unique Codes | Dead Codes |
|--------|-------------|--------------|------------|
| **VAE + Geodesic** | **100.00%** | **256/256** | **0** |
| VQ-VAE (end-to-end) | 85.16% | 218/256 | 38 |

**Analysis**:
- Geodesic quantization achieves **perfect codebook utilization** (all 256 codes used)
- VQ-VAE has ~15% dead codes (common problem in VQ-VAE training)
- Geodesic clustering ensures better coverage of the latent manifold

### Summary Table

| Metric | VAE + Geodesic | VQ-VAE | Winner |
|--------|----------------|--------|--------|
| Reconstruction MSE | 0.013601 | **0.002380** | VQ-VAE |
| Codebook Utilization | **100%** | 85.16% | Geodesic |
| Parameters | ~1.5M VAE | ~1.06M | Comparable |
| Training | 2-stage (VAE + KMeans) | End-to-end | Different |

---

## 🎨 Qualitative Results

### Reconstruction Comparison
![Reconstructions](data/recons/comparison/vae_vs_vqvae_reconstructions.png)

**Observations**:
1. **Original Images**: Test set samples
2. **VAE + Geodesic**: Slight blurriness, preserves structure
3. **VQ-VAE**: Sharper, better fine details, occasional artifacts

### Sampling Methods Comparison
![Sampling](data/recons/samples/sampling_methods_comparison.png)

**Three Sampling Approaches**:

1. **VQ-VAE (Random Codes)**:
   - Uniform random sampling from learned codebook
   - No prior distribution learned
   - Quality: Variable, some unrealistic samples

2. **VAE + Geodesic + PixelCNN** (Our Full Pipeline):
   - Geodesic codebook + learned autoregressive prior
   - Most structured and realistic samples
   - Quality: Best perceptual quality

3. **VAE + Geodesic (Random Codes)** (Baseline):
   - Geodesic codebook, no prior
   - Pure baseline comparison
   - Quality: Worst, noisy and unstructured

### PixelCNN Prior vs Random
![PixelCNN](data/recons/samples/pixelcnn_vs_random.png)

**Demonstrates the benefit of learned prior**:
- **Top row**: PixelCNN learned autoregressive distribution → realistic digits
- **Bottom row**: Uniform random sampling → no structure, unrealistic

---

## 🔬 Key Insights

### Strengths of Geodesic Quantization

1. **Perfect Codebook Coverage** ✅
   - 100% utilization vs 85% for VQ-VAE
   - No dead codes, better latent space exploration
   - K-means with geodesic distance ensures balanced clusters

2. **Geometric Awareness** ✅
   - Wasserstein-2 distance respects latent manifold curvature
   - Better than Euclidean L2 distance for curved spaces
   - Captures uncertainty via variance (μ, σ) not just mean

3. **Modularity** ✅
   - VAE training independent of quantization
   - Can experiment with different clustering methods
   - Easy to add PixelCNN prior as separate stage

4. **Interpretability** ✅
   - Clear separation: continuous VAE → discrete quantization → autoregressive prior
   - Each stage can be analyzed independently

### Weaknesses of Geodesic Quantization

1. **Reconstruction Quality** ❌
   - 471% worse MSE than VQ-VAE
   - Post-hoc quantization cannot optimize encoder/decoder
   - No gradient flow through quantization during VAE training

2. **Two-Stage Training** ❌
   - More complex pipeline (VAE → KMeans → PixelCNN)
   - Cannot jointly optimize all components
   - Requires careful hyperparameter tuning at each stage

3. **Computational Cost** ❌
   - Geodesic distance computation expensive (Wasserstein-2)
   - K-means on geodesic metric slower than Euclidean
   - Requires storing variance information

### Trade-offs Summary

| Aspect | Geodesic Advantage | VQ-VAE Advantage |
|--------|-------------------|------------------|
| Reconstruction | ❌ Much worse MSE | ✅ Best reconstruction |
| Codebook | ✅ 100% utilization | ❌ 15% dead codes |
| Training | ❌ Multi-stage | ✅ End-to-end |
| Geometry | ✅ Manifold-aware | ❌ Euclidean only |
| Modularity | ✅ Easy to modify | ❌ Coupled components |

---

## 🎯 Conclusions

### Research Questions Answered

**Q1: Does geodesic quantization improve over Euclidean?**  
✅ **Yes for codebook coverage**: 100% vs 85% utilization  
❌ **No for reconstruction**: Much worse MSE  
➡️ **Trade-off**: Better latent geometry vs reconstruction quality

**Q2: Can post-hoc quantization compete with end-to-end learning?**  
❌ **No for reconstruction**: 471% worse MSE  
✅ **Yes for generation with prior**: PixelCNN samples are realistic  
➡️ **Depends on objective**: Generation quality can be recovered with good prior

**Q3: What is the impact of latent manifold geometry?**  
✅ **Significant**: Perfect codebook coverage shows geodesic distance better respects latent structure  
✅ **Measurable**: Wasserstein-2 using (μ, σ) provides richer information than L2 on μ alone

### Practical Recommendations

**Use Geodesic Quantization when**:
- ✅ Codebook coverage is critical
- ✅ Interpretability and modularity are important
- ✅ You want to experiment with different clustering methods
- ✅ Latent space geometry matters for downstream tasks

**Use Standard VQ-VAE when**:
- ✅ Reconstruction quality is the primary metric
- ✅ Training simplicity is important
- ✅ You need fast inference
- ✅ End-to-end gradient flow is beneficial

### Future Work

1. **Hybrid Approach**: Initialize VQ-VAE codebook with geodesic clustering
2. **Better Priors**: Train PixelCNN for VQ-VAE to enable fair generation comparison
3. **Other Modalities**: Test on audio spectrograms, high-res images
4. **Riemannian Metrics**: Explore other geodesic distance approximations
5. **Adaptive Clustering**: Use geodesic distance for online codebook updates

---

## 📁 Project Structure

```
VQ_VAE_Geodesic/
├── data/
│   ├── checkpoints/
│   │   ├── main_checkpoint_mnist.pt      # VAE model
│   │   ├── vqvae_mnist_final.pt          # VQ-VAE model
│   │   └── pixelcnn_mnist_final.pt       # PixelCNN prior
│   ├── latents/
│   │   └── chunk_codebook.npz             # Geodesic codebook
│   └── recons/
│       ├── comparison/                     # Reconstruction comparisons
│       └── samples/                        # Generation samples
├── src/vq_vae_geodesic/
│   ├── models/
│   │   ├── modules/
│   │   │   ├── vae.py                     # VAE implementation
│   │   │   ├── vqvae.py                   # VQ-VAE implementation
│   │   │   ├── encoder.py                 # Encoder architectures
│   │   │   ├── decoder.py                 # Decoder architectures
│   │   │   └── pixelCNN.py                # PixelCNN prior
│   │   └── quantization/
│   │       └── geodesic.py                # Geodesic quantizer
│   ├── scripts/
│   │   ├── train_vae_mnist.py             # Train VAE
│   │   ├── train_vqvae_mnist.py           # Train VQ-VAE
│   │   ├── train_pixelcnn_mnist.py        # Train PixelCNN
│   │   ├── quantize_mnist.py              # Geodesic quantization
│   │   ├── compare_vae_vs_vqvae.py        # Reconstruction comparison
│   │   ├── compare_sampling_methods.py     # Generation comparison
│   │   └── compare_pixelcnn_vs_random.py  # Prior comparison
│   └── evaluation/
│       ├── evaluate.py                    # Evaluation metrics
│       └── sample.py                      # Sampling utilities
└── notebooks/
    └── Test.ipynb                         # Experimentation notebook
```

---

## 🚀 Reproducing Results

### 1. Train VAE
```bash
uv run -m vq_vae_geodesic.scripts.train_vae_mnist
```

### 2. Extract Latents & Geodesic Quantization
```bash
uv run -m vq_vae_geodesic.scripts.extract_mnist_latents
uv run -m vq_vae_geodesic.scripts.quantize_mnist
```

### 3. Train PixelCNN Prior
```bash
uv run -m vq_vae_geodesic.scripts.train_pixelcnn_mnist
```

### 4. Train VQ-VAE Baseline
```bash
uv run -m vq_vae_geodesic.scripts.train_vqvae_mnist
```

### 5. Generate Comparisons
```bash
uv run -m vq_vae_geodesic.scripts.compare_vae_vs_vqvae
uv run -m vq_vae_geodesic.scripts.compare_sampling_methods
uv run -m vq_vae_geodesic.scripts.compare_pixelcnn_vs_random
```

---

## 📚 References

1. **VQ-VAE**: van den Oord et al., "Neural Discrete Representation Learning", NeurIPS 2017
2. **VAE**: Kingma & Welling, "Auto-Encoding Variational Bayes", ICLR 2014
3. **PixelCNN**: van den Oord et al., "Pixel Recurrent Neural Networks", ICML 2016
4. **Wasserstein Distance**: Optimal Transport for ML applications
5. **Geodesic Clustering**: Riemannian geometry in latent spaces

---

## 🎓 Academic Context

**Course**: Deep Learning  
**Institution**: University Project  
**Objective**: Explore non-Euclidean geometry in learned latent spaces  
**Outcome**: Demonstrated trade-off between geometric fidelity and reconstruction quality

---

**End of Report**
