import torch
from tqdm import tqdm
def extract_latents(model, dataloader, device):
    model.eval()
    mus = []
    logvars = []
    with torch.no_grad():
        for image_batch, _ in tqdm(dataloader, desc="Extracting latents"):
            image_batch = image_batch.to(device)
            mu, logvar = model.encoder(image_batch)
            mus.append(mu.cpu())
            logvars.append(logvar.cpu())
    mus = torch.cat(mus, dim=0)
    logvars = torch.cat(logvars, dim=0)
    return mus, logvars

def save_latents(mus, logvars, save_path):
    torch.save({'mu': mus, 'logvar': logvars}, save_path)
    print(f"Latent representations saved to {save_path} (mu shape: {mus.shape}, logvar shape: {logvars.shape})")
    return save_path

def extract_and_save_latents(model, dataloader, device, save_path):
    mus, logvars = extract_latents(model, dataloader, device)
    return save_latents(mus, logvars, save_path)