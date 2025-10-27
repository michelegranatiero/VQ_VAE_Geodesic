import torch
from vq_vae_geodesic.config import checkpoint_dir, latents_dir
from vq_vae_geodesic.models.modules.vae import build_vae_from_config
from vq_vae_geodesic.models.modules.vqvae import build_vqvae_from_config
from vq_vae_geodesic.models.modules.pixelCNN import build_pixelcnn_from_config
import numpy as np

# ----------------------------------------------
# ------------ VAE ( + Geodesic) ---------------
# ----------------------------------------------
def load_model_vae_mnist(arch_params, device):
	"""
	Load VAE model for MNIST from checkpoint.
	"""
	checkpoint_path = checkpoint_dir('mnist') / "checkpoint_mnist_best.pt"
	model = build_vae_from_config(arch_params)
	checkpoint = torch.load(checkpoint_path, map_location=device)
	model.load_state_dict(checkpoint['model_state_dict'])
	model = model.to(device)
	model.eval()
	return model

def load_latents_mnist(latents_path, map_location='cpu'):
	"""
	Load mu and logvar tensors from a .pt file (es. train_latents.pt).
	Returns: mu, logvar
	"""
	if not latents_path.exists():
		raise FileNotFoundError(
			f"Latents not found at {latents_path}\n"
			"Run extraction first"
		)
	data = torch.load(latents_path, map_location=map_location)
	return data['mu'], data['logvar']

def load_codebook_mnist(device):
	"""
	Load codebook_chunks tensor for MNIST from file.
	"""
	codebook_path = latents_dir('mnist') / "chunk_codebook.pt"
	if not codebook_path.exists():
		raise FileNotFoundError(
			f"Codebook not found at {codebook_path}\n"
			"Run quantization first: python -m vq_vae_geodesic.scripts.quantize_mnist"
		)
	codebook_data = torch.load(codebook_path, map_location=device)
	return codebook_data['codebook_chunks']

def load_codes_indices_mnist():
	"""
	Load train, val, test codes (indices) for MNIST from file.
	"""
	codes_path = latents_dir('mnist') / "assigned_codes.pt"
	if not codes_path.exists():
		raise FileNotFoundError(
			f"Codes not found at {codes_path}\n"
			"Run quantization first: python -m vq_vae_geodesic.scripts.quantize_mnist"
		)
	codes_data = torch.load(codes_path, map_location="cpu")
	return codes_data['train_codes'], codes_data['val_codes'], codes_data['test_codes']

# Get actual discrete codes from indices
def lookup_codewords(codebook_chunks: torch.Tensor, batch_codes: torch.Tensor) -> torch.Tensor:
	"""
	Given a codebook and a batch of code indices, returns the quantized latent vectors.
	Args:
		codebook_chunks: (K, chunk_size) tensor
		batch_codes: (bs, L) tensor of indices
	Returns:
		z_recon: (bs, D) tensor, where D = L * chunk_size
	"""
	z_chunks = codebook_chunks[batch_codes]  # indexing
	z_recon = z_chunks.reshape(z_chunks.size(0), -1)
	return z_recon

# ----------------------------------------
# ----------------- VQ-VAE ---------------
# ----------------------------------------
def load_model_vqvae_mnist(arch_params, vqvae_params, device):
	"""
	Load VQ-VAE model for MNIST from checkpoint.
	"""
	checkpoint_path = checkpoint_dir('mnist') / "vqvae_mnist_best.pt"
	model = build_vqvae_from_config(arch_params, vqvae_params)
	checkpoint = torch.load(checkpoint_path, map_location=device)
	model.load_state_dict(checkpoint['model_state_dict'])
	model = model.to(device)
	model.eval()
	return model

 #  codes extraction and saving 
def extract_vqvae_codes(vqvae, data_loader, device):
	"""
	Extract discrete codes from a VQ-VAE model for all images in a dataloader.
	Returns: np.ndarray of codes (N, H, W)
	"""
	from tqdm import tqdm
	vqvae.eval()
	all_codes = []
	with torch.no_grad():
		for x, _ in tqdm(data_loader, desc="Extracting VQ-VAE codes"):
			x = x.to(device)
			z_e = vqvae.encoder(x)
			_, _, codes = vqvae.vq(z_e)
			all_codes.append(codes.cpu().numpy())
	all_codes = np.concatenate(all_codes, axis=0)
	return all_codes

def save_codes(codes, save_path):
	if isinstance(codes, np.ndarray):
		codes = torch.from_numpy(codes)
	torch.save({'codes': codes}, save_path)
	print(f"Codes saved to {save_path} (shape: {codes.shape})")
	return save_path

# ----------------------------------------
# ------------ PixelCNN -------------------
# ----------------------------------------
def load_pixelcnn_checkpoint(checkpoint_name, config, device, is_vqvae=False):

	# Determine dataset from config
	dataset = 'cifar10' if 'cifar' in checkpoint_name.lower() else 'mnist'
	path = checkpoint_dir(dataset) / checkpoint_name
	if not path.exists():
		raise FileNotFoundError(f"PixelCNN checkpoint not found: {path}")
	model = build_pixelcnn_from_config(config, for_vqvae=is_vqvae)
	
	checkpoint = torch.load(path, map_location=device)
	model.load_state_dict(checkpoint['model_state_dict'])
	model = model.to(device)
	model.eval()
	return model

def codes_to_images_via_codebook(codes, codebook_chunks, decoder, device):
	"""
	Convert discrete codes to images via codebook lookup and VAE decoder.
	
	Args:
		codes: Discrete codes (B, H, W) as numpy array or torch tensor
		codebook_chunks: Codebook tensor (K, chunk_size)
		decoder: VAE decoder model
		device: Device for computation
		
	Returns:
		images: Generated images as numpy array (B, C, H, W)
	"""
	if isinstance(codes, np.ndarray):
		batch_size = codes.shape[0]
		codes_flat = torch.from_numpy(codes.reshape(batch_size, -1)).long().to(device)
	else:
		batch_size = codes.shape[0]
		codes_flat = codes.reshape(batch_size, -1).long().to(device)
	
	# Use lookup_codewords to convert indices to latents
	latents = lookup_codewords(codebook_chunks, codes_flat)
	
	# Decode to images
	decoder.eval()
	with torch.no_grad():
		images = decoder(latents)
	
	return images.cpu().numpy()

def codes_to_images_via_vqvae(codes, vqvae, device):
	"""
	Convert discrete codes to images via VQ-VAE embeddings and decoder.
	
	Args:
		codes: Discrete codes (B, H, W) as numpy array or torch tensor
		vqvae: VQ-VAE model with vq.embeddings and decoder
		device: Device for computation
		
	Returns:
		images: Generated images as numpy array (B, C, H, W)
	"""
	if isinstance(codes, np.ndarray):
		codes_t = torch.from_numpy(codes).long().to(device)
	else:
		codes_t = codes.long().to(device)
	
	# Map codes to embeddings
	embeddings = vqvae.vq.embeddings.weight
	quantized_latents = embeddings[codes_t]  # (B, H, W, embed_dim)
	quantized_latents = quantized_latents.permute(0, 3, 1, 2).contiguous()  # (B, embed_dim, H, W)
	
	# Decode
	vqvae.eval()
	with torch.no_grad():
		images = vqvae.decoder(quantized_latents)
	
	return images.cpu().numpy()


# ----------------------------------------------
# ------------ CIFAR-10 Functions --------------
# ----------------------------------------------

def load_model_vae_cifar10(arch_params, device):
	"""
	Load VAE model for CIFAR-10 from checkpoint.
	"""
	checkpoint_path = checkpoint_dir('cifar10') / "vae_cifar10_best.pt"
	model = build_vae_from_config(arch_params, dataset="cifar")
	checkpoint = torch.load(checkpoint_path, map_location=device)
	model.load_state_dict(checkpoint['model_state_dict'])
	model = model.to(device)
	model.eval()
	return model

def load_latents_cifar10(latents_path, map_location='cpu'):
	"""
	Load mu and logvar tensors from a .pt file (e.g. train_latents_cifar10.pt).
	Returns: mu, logvar
	"""
	if not latents_path.exists():
		raise FileNotFoundError(
			f"Latents not found at {latents_path}\n"
			"Run extraction first"
		)
	data = torch.load(latents_path, map_location=map_location)
	return data['mu'], data['logvar']

def load_codebook_cifar10(device):
	"""
	Load codebook_chunks tensor for CIFAR-10 from file.
	"""
	codebook_path = latents_dir('cifar10') / "chunk_codebook_cifar10.pt"
	if not codebook_path.exists():
		raise FileNotFoundError(
			f"Codebook not found at {codebook_path}\n"
			"Run quantization first: python -m vq_vae_geodesic.scripts.quantize_cifar10"
		)
	codebook_data = torch.load(codebook_path, map_location=device)
	return codebook_data['codebook_chunks']

def load_codes_indices_cifar10():
	"""
	Load train, val, test codes (indices) for CIFAR-10 from file.
	"""
	codes_path = latents_dir('cifar10') / "assigned_codes_cifar10.pt"
	if not codes_path.exists():
		raise FileNotFoundError(
			f"Codes not found at {codes_path}\n"
			"Run quantization first: python -m vq_vae_geodesic.scripts.quantize_cifar10"
		)
	codes_data = torch.load(codes_path, map_location="cpu")
	return codes_data['train_codes'], codes_data['val_codes'], codes_data['test_codes']

def load_model_vqvae_cifar10(arch_params, vqvae_params, device):
	"""
	Load VQ-VAE model for CIFAR-10 from checkpoint.
	"""
	checkpoint_path = checkpoint_dir('cifar10') / "vqvae_cifar10_best.pt"
	model = build_vqvae_from_config(arch_params, vqvae_params, dataset="cifar")
	checkpoint = torch.load(checkpoint_path, map_location=device)
	model.load_state_dict(checkpoint['model_state_dict'])
	model = model.to(device)
	model.eval()
	return model


# ----------------------------------------------
# ------------ CelebA Functions ----------------
# ----------------------------------------------

def load_model_vae_celeba(arch_params, device):
	"""
	Load VAE model for CelebA from checkpoint.
	Uses same RGB architecture as CIFAR-10 (32×32 only).
	"""
	checkpoint_path = checkpoint_dir('celeba') / "vae_celeba_best.pt"
	model = build_vae_from_config(arch_params, dataset="celeba")
	checkpoint = torch.load(checkpoint_path, map_location=device)
	model.load_state_dict(checkpoint['model_state_dict'])
	model = model.to(device)
	model.eval()
	return model

def load_latents_celeba(latents_path, map_location='cpu'):
	"""
	Load mu and logvar tensors from a .pt file (e.g. train_latents_celeba.pt).
	Returns: mu, logvar
	"""
	if not latents_path.exists():
		raise FileNotFoundError(
			f"Latents not found at {latents_path}\n"
			"Run extraction first"
		)
	data = torch.load(latents_path, map_location=map_location)
	return data['mu'], data['logvar']

def load_codebook_celeba(device):
	"""
	Load codebook_chunks tensor for CelebA from file.
	"""
	codebook_path = latents_dir('celeba') / "chunk_codebook_celeba.pt"
	if not codebook_path.exists():
		raise FileNotFoundError(
			f"Codebook not found at {codebook_path}\n"
			"Run quantization first: python -m vq_vae_geodesic.scripts_celeba.quantize_celeba"
		)
	codebook_data = torch.load(codebook_path, map_location=device)
	return codebook_data['codebook_chunks']

def load_codes_indices_celeba():
	"""
	Load train, val, test codes (indices) for CelebA from file.
	"""
	codes_path = latents_dir('celeba') / "assigned_codes_celeba.pt"
	if not codes_path.exists():
		raise FileNotFoundError(
			f"Codes not found at {codes_path}\n"
			"Run quantization first: python -m vq_vae_geodesic.scripts_celeba.quantize_celeba"
		)
	codes_data = torch.load(codes_path, map_location="cpu")
	return codes_data['train_codes'], codes_data['val_codes'], codes_data['test_codes']

def load_model_vqvae_celeba(arch_params, vqvae_params, device):
	"""
	Load VQ-VAE model for CelebA from checkpoint.
	Uses same RGB architecture as CIFAR-10.
	"""
	checkpoint_path = checkpoint_dir('celeba') / "vqvae_celeba_best.pt"
	model = build_vqvae_from_config(arch_params, vqvae_params, dataset="cifar")  # RGB architecture
	checkpoint = torch.load(checkpoint_path, map_location=device)
	model.load_state_dict(checkpoint['model_state_dict'])
	model = model.to(device)
	model.eval()
	return model