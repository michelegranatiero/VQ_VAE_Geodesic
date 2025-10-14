import torch
from tqdm import tqdm
import wandb
from vq_vae_geodesic.utils import make_averager, refresh_bar
import torch.nn.functional as F


def step(model, loss_fn, xb, yb, device, variational_beta, opt=None):
    image_batch = xb.to(device)
    target_batch = image_batch  # For autoencoders, target is the input itself
    image_batch_recon, latent_mu, latent_logvar = model(image_batch)
    loss = loss_fn(image_batch_recon, target_batch, latent_mu, latent_logvar, beta=variational_beta)

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        opt.step()

    return loss, image_batch_recon, latent_mu, latent_logvar


def fit_vae(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    variational_beta,
    num_epochs,
    device,
    start_epoch=1,
    checkpoint_path=None,
    train_loss_history=None,
    val_loss_history=None,
    save_checkpoint_every=1     # Save checkpoint every N epochs
):
    # Initialize loss histories if not provided
    train_loss_avg = train_loss_history if train_loss_history is not None else []
    val_loss_avg = val_loss_history if val_loss_history is not None else []

    fixed_batch = None
    model = model.to(device)

    print(f'Training from epoch {start_epoch} to {num_epochs}...')

    for epoch in range(start_epoch, num_epochs + 1):
        # --- TRAIN ---
        model.train()
        train_loss_averager = make_averager()
        batch_bar = tqdm(train_loader, leave=False, desc=f'epoch {epoch} train', total=len(train_loader))

        for image_batch, target_batch in batch_bar:
            loss, *_ = step(
                model, loss_fn, image_batch, target_batch, device,
                variational_beta, optimizer
            )
            train_loss_averager(loss.item())
            refresh_bar(batch_bar, f"train batch [loss: {train_loss_averager(None):.3f}]")

        train_loss_avg.append(train_loss_averager(None))

        # --- VALIDATION ---
        model.eval()
        val_loss_averager = make_averager()
        with torch.no_grad():
            val_bar = tqdm(val_loader, total=len(val_loader), desc=f'epoch {epoch} validation')
            for image_batch, target_batch in val_bar:
                loss, *_ = step(
                    model, loss_fn, image_batch, target_batch, device,
                    variational_beta, opt=None
                )
                val_loss_averager(loss.item())
                refresh_bar(val_bar, f"validation batch [loss: {val_loss_averager(None):.3f}]")

            val_loss_avg.append(val_loss_averager(None))

        print(
            f"Epoch: {epoch}\n"
            f"Train set: Average loss: {train_loss_avg[-1]:.4f}\n"
            f"Validation set: Average loss: {val_loss_avg[-1]:.4f}\n"
        )

        # Logging wandb
        with torch.no_grad():
            if fixed_batch is None:
                sample_batch = next(iter(val_loader))
                fixed_batch = (sample_batch[0][:4].clone(), sample_batch[1][:4].clone())

            fixed_images, fixed_targets = fixed_batch
            fixed_images = fixed_images.to(device)
            fixed_targets = fixed_images  # For autoencoders, target is the input itself
            fixed_recon, *_ = model(fixed_images)

            original = fixed_images[0].cpu()
            reconstruction = fixed_recon[0].cpu()

            comparison = torch.cat([original, reconstruction], dim=2)

            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss_avg[-1],
                "val_loss": val_loss_avg[-1],
                "comparison": wandb.Image(
                    comparison,
                    caption=f"Left: Original, Right: Reconstructed (Epoch {epoch})"
                )
            })

        # Save checkpoint
        if checkpoint_path and (epoch % save_checkpoint_every == 0 or epoch == num_epochs):
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss_history': train_loss_avg,
                'val_loss_history': val_loss_avg,
                'variational_beta': variational_beta,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch}")
            wandb.save(checkpoint_path)

    return train_loss_avg, val_loss_avg


def fit_pixelcnn(
    model,
    train_loader,
    val_loader,
    optimizer,
    num_epochs,
    device,
    start_epoch=1,
    checkpoint_path=None,
    train_loss_history=None,
    val_loss_history=None,
    save_checkpoint_every=1
):
    """
    Train PixelCNN autoregressive prior on discrete latent codes.

    Args:
        model: PixelCNN model
        train_loader: DataLoader with code grids (B, H, W)
        val_loader: Validation DataLoader (optional)
        optimizer: Optimizer for training
        num_epochs: Number of epochs to train
        device: Device to use
        start_epoch: Starting epoch (for resuming)
        checkpoint_path: Path to save checkpoints
        train_loss_history: List to track training loss
        val_loss_history: List to track validation loss
        save_checkpoint_every: Save frequency

    Returns:
        train_loss_history, val_loss_history
    """

    # Initialize loss histories
    train_loss_avg = train_loss_history if train_loss_history is not None else []
    val_loss_avg = val_loss_history if val_loss_history is not None else []

    model = model.to(device)
    best_val_loss = float('inf')

    print(f'Training PixelCNN from epoch {start_epoch} to {num_epochs}...')

    for epoch in range(start_epoch, num_epochs + 1):
        # --- TRAIN ---
        model.train()
        train_loss_averager = make_averager()
        batch_bar = tqdm(train_loader, leave=False, desc=f'epoch {epoch} train', total=len(train_loader))

        for codes_batch in batch_bar:
            codes_batch = codes_batch.to(device)  # (B, H, W)

            optimizer.zero_grad()
            logits = model(codes_batch)  # (B, K, H, W)
            loss = F.cross_entropy(logits, codes_batch)
            loss.backward()
            optimizer.step()

            train_loss_averager(loss.item())
            refresh_bar(batch_bar, f"train batch [loss: {train_loss_averager(None):.3f}]")

        train_loss_avg.append(train_loss_averager(None))

        # --- VALIDATION ---
        model.eval()
        val_loss_averager = make_averager()
        with torch.no_grad():
            val_bar = tqdm(val_loader, total=len(val_loader), desc=f'epoch {epoch} validation')
            for codes_batch in val_bar:
                codes_batch = codes_batch.to(device)
                logits = model(codes_batch)
                loss = F.cross_entropy(logits, codes_batch)
                val_loss_averager(loss.item())
                refresh_bar(val_bar, f"validation batch [loss: {val_loss_averager(None):.3f}]")

        val_loss_avg.append(val_loss_averager(None))

        print(
            f"Epoch: {epoch}\n"
            f"Train set: Average loss: {train_loss_avg[-1]:.4f}\n"
            f"Validation set: Average loss: {val_loss_avg[-1]:.4f}\n"
        )

        # Save best model
        if val_loss_avg[-1] < best_val_loss:
            best_val_loss = val_loss_avg[-1]
            if checkpoint_path:
                best_path = str(checkpoint_path).replace('.pt', '_best.pt')
                torch.save(model.state_dict(), best_path)
                print(f"Best model saved with val loss: {best_val_loss:.4f}")

        # logging wandb
        log_dict = {
            "epoch": epoch,
            "train_loss": train_loss_avg[-1],
        }
        if val_loader is not None:
            log_dict["val_loss"] = val_loss_avg[-1]
        wandb.log(log_dict)

        # Save checkpoint
        if checkpoint_path and (epoch % save_checkpoint_every == 0 or epoch == num_epochs):
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss_history': train_loss_avg,
                'val_loss_history': val_loss_avg,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch}")
            try:
                wandb.save(checkpoint_path)
            except:
                pass  # wandb not initialized or disabled

    return train_loss_avg, val_loss_avg


def fit_vqvae(
    model,
    train_loader,
    val_loader,
    optimizer,
    num_epochs,
    device,
    start_epoch=1,
    checkpoint_path=None,
    train_loss_history=None,
    val_loss_history=None,
    save_checkpoint_every=1
):
    # Initialize loss histories
    train_loss_avg = train_loss_history if train_loss_history is not None else []
    val_loss_avg = val_loss_history if val_loss_history is not None else []

    model = model.to(device)
    best_val_loss = float('inf')

    print(f"Training VQ-VAE from epoch {start_epoch} to {num_epochs}...")

    for epoch in range(start_epoch, num_epochs + 1):
        # --- TRAIN ---
        model.train()
        train_loss_averager = make_averager()
        train_recon_averager = make_averager()
        train_vq_averager = make_averager()
        batch_bar = tqdm(train_loader, leave=False, desc=f'epoch {epoch} train', total=len(train_loader))

        for image_batch, _ in batch_bar:
            image_batch = image_batch.to(device)
            optimizer.zero_grad()

            # Forward pass
            recon, vq_loss, _ = model(image_batch)

            # Reconstruction loss (MSE)
            recon_loss = F.mse_loss(recon, image_batch)

            # Total loss
            loss = recon_loss + vq_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss_averager(loss.item())
            train_recon_averager(recon_loss.item())
            train_vq_averager(vq_loss.item())
            refresh_bar(
                batch_bar, f"train batch [loss: {train_loss_averager(None):.3f} | recon: {train_recon_averager(None):.3f} | vq: {train_vq_averager(None):.3f}]")

        train_loss_avg.append(train_loss_averager(None))
        train_recon_avg = train_recon_averager(None)
        train_vq_avg = train_vq_averager(None)

        # --- VALIDATION ---
        model.eval()
        val_loss_averager = make_averager()
        val_recon_averager = make_averager()
        val_vq_averager = make_averager()
        with torch.no_grad():
            val_bar = tqdm(val_loader, total=len(val_loader), desc=f'epoch {epoch} validation')
            for image_batch, _ in val_bar:
                image_batch = image_batch.to(device)
                recon, vq_loss, _ = model(image_batch)
                recon_loss = F.mse_loss(recon, image_batch)
                loss = recon_loss + vq_loss
                val_loss_averager(loss.item())
                val_recon_averager(recon_loss.item())
                val_vq_averager(vq_loss.item())
                refresh_bar(
                    val_bar, f"validation batch [loss: {val_loss_averager(None):.3f} | recon: {val_recon_averager(None):.3f} | vq: {val_vq_averager(None):.3f}]")

            val_loss_avg.append(val_loss_averager(None))
            val_recon_avg = val_recon_averager(None)
            val_vq_avg = val_vq_averager(None)

        print(
            f"Epoch {epoch}: "
            f"Train Loss={train_loss_avg[-1]:.4f} (recon={train_recon_avg:.4f}, vq={train_vq_avg:.4f}) | "
            f"Val Loss={val_loss_avg[-1]:.4f} (recon={val_recon_avg:.4f}, vq={val_vq_avg:.4f})"
        )

        # Save best model
        if val_loss_avg[-1] < best_val_loss:
            best_val_loss = val_loss_avg[-1]
            if checkpoint_path:
                best_path = checkpoint_path.replace('.pt', '_best.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss
                }, best_path)
                print(f"Best model saved (val_loss={best_val_loss:.4f})")

        # logging wandb
        with torch.no_grad():
            log_dict = {
                "epoch": epoch,
                "train_loss": train_loss_avg[-1],
                "train_recon_loss": train_recon_avg,
                "train_vq_loss": train_vq_avg,
            }
            log_dict["val_loss"] = val_loss_avg[-1]
            log_dict["val_recon_loss"] = val_recon_avg
            log_dict["val_vq_loss"] = val_vq_avg

            # Logging immagini ricostruite (come in fit_vae)
            if epoch == start_epoch:
                # Prendi un batch fisso dalla validation
                sample_batch = next(iter(val_loader))
                fixed_images = sample_batch[0][:4].clone()

                fixed_images = fixed_images.to(device)
                fixed_recon, _, _ = model(fixed_images)
                original = fixed_images[0].cpu()
                reconstruction = fixed_recon[0].cpu()
                comparison = torch.cat([original, reconstruction], dim=2)
                log_dict["comparison"] = wandb.Image(
                    comparison,
                    caption=f"Left: Original, Right: Reconstructed (Epoch {epoch})"
                )
            wandb.log(log_dict)

        # Save checkpoint
        if checkpoint_path and (epoch % save_checkpoint_every == 0 or epoch == num_epochs):
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss_history': train_loss_avg,
                'val_loss_history': val_loss_avg,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch}")
            try:
                wandb.save(checkpoint_path)
            except:
                pass  # wandb not initialized or disabled

    return train_loss_avg, val_loss_avg
