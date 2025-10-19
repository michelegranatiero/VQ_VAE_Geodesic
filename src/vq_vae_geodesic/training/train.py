import torch
from tqdm import tqdm
import wandb
from vq_vae_geodesic.utils import make_averager, refresh_bar
import torch.nn.functional as F
from vq_vae_geodesic.metrics import compute_image_metrics
from vq_vae_geodesic.utils import EarlyStopping


def step_vae(model, loss_fn, xb, device, opt=None):
    image_batch = xb.to(device)
    target_batch = image_batch  # For autoencoders, target is the input itself
    x_recon, latent_mu, latent_logvar = model(image_batch)
    loss, recon_loss = loss_fn(x_recon, target_batch, latent_mu, latent_logvar)

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        opt.step()

    return loss.item(), recon_loss.item(), x_recon.detach(), latent_mu.detach(), latent_logvar.detach()


def fit_vae(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    num_epochs,
    device,
    checkpoint_path,
    start_epoch=1,
    train_loss_history=None,
    train_recon_history=None,
    val_loss_history=None,
    val_recon_history=None,
    save_checkpoint_every=1     # Save checkpoint every N epochs
):
    # Initialize loss histories if not provided
    train_loss_avg_history = train_loss_history if train_loss_history is not None else []
    train_recon_avg_history = train_recon_history if train_recon_history is not None else []
    val_loss_avg_history = val_loss_history if val_loss_history is not None else []
    val_recon_avg_history = val_recon_history if val_recon_history is not None else []

    fixed_batch = None  # To store a fixed batch for reconstruction visualization
    model = model.to(device)

    print(f'Training from epoch {start_epoch} to {num_epochs}...')

    # Path per best model (early stopping)
    checkpoint_best_path = str(checkpoint_path).replace('.pt', '_best.pt') if checkpoint_path else None
    es = EarlyStopping(patience=5, min_delta=0.0, mode='min', path=checkpoint_best_path)

    for epoch in range(start_epoch, num_epochs + 1):
        # --- TRAIN ---
        model.train()
        train_loss_avg = make_averager()
        train_recon_avg = make_averager()

        batch_bar = tqdm(train_loader, leave=False, desc=f'epoch {epoch} train', total=len(train_loader))
        for image_batch, _ in batch_bar:
            loss, recon_loss, x_recon, latent_mu, latent_logvar = step_vae(
                model, loss_fn, image_batch, device, optimizer
            )
            train_loss_avg(loss)
            train_recon_avg(recon_loss)
            refresh_bar(
                batch_bar, f"train batch [loss: {train_loss_avg(None):.3f} | recon loss: {train_recon_avg(None):.3f}]")

        train_loss_avg_history.append(train_loss_avg(None))
        train_recon_avg_history.append(train_recon_avg(None))

        # --- VALIDATION ---
        model.eval()
        val_loss_avg = make_averager()
        val_recon_avg = make_averager()

        val_l1_avg = make_averager()
        val_mse_avg = make_averager()
        val_psnr_avg = make_averager()
        val_ssim_avg = make_averager()

        with torch.no_grad():
            val_bar = tqdm(val_loader, total=len(val_loader), desc=f'epoch {epoch} validation')
            for image_batch, _ in val_bar:
                loss, recon_loss, x_recon, latent_mu, latent_logvar = step_vae(
                    model, loss_fn, image_batch, device, opt=None
                )
                val_loss_avg(loss)
                val_recon_avg(recon_loss)

                metrics = compute_image_metrics(x_recon, image_batch.to(device))
                val_l1_avg(metrics['l1'])
                val_mse_avg(metrics['mse'])
                val_psnr_avg(metrics['psnr'])
                val_ssim_avg(metrics['ssim'])

                refresh_bar(
                    val_bar, f"validation batch [loss: {val_loss_avg(None):.3f} | recon loss: {val_recon_avg(None):.3f}]")

            val_loss_avg_history.append(val_loss_avg(None))
            val_recon_avg_history.append(val_recon_avg(None))

        print(
            f"Epoch: {epoch}\n"
            f"Train Loss={train_loss_avg_history[-1]:.4f} (recon loss={train_recon_avg_history[-1]:.4f})\n"
            f"Validation Loss={val_loss_avg_history[-1]:.4f} (recon loss={val_recon_avg_history[-1]:.4f})\n"
        )

        # Logging wandb (and visualization)
        with torch.no_grad():
            if fixed_batch is None:
                sample_batch = next(iter(val_loader))
                fixed_batch = sample_batch[0][:4].clone() # first 4 images

            fixed_images = fixed_batch.to(device)
            fixed_recon, *_ = model(fixed_images)

            originals = fixed_images.cpu()
            reconstructions = fixed_recon.cpu()

            # 2x2 grid for originals
            orig_top = torch.cat([originals[0], originals[1]], dim=2)  # horizontal concat
            orig_bottom = torch.cat([originals[2], originals[3]], dim=2)
            grid_original = torch.cat([orig_top, orig_bottom], dim=1)  # vertical concat

            # 2x2 grid for reconstructions
            recon_top = torch.cat([reconstructions[0], reconstructions[1]], dim=2)
            recon_bottom = torch.cat([reconstructions[2], reconstructions[3]], dim=2)
            grid_recon = torch.cat([recon_top, recon_bottom], dim=1)

            # Concatenate original and reconstructed grids
            comparison = torch.cat([grid_original, grid_recon], dim=2)  # horizontal concat

            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss_avg_history[-1],
                "train/recon_loss": train_recon_avg_history[-1],
                "val/loss": val_loss_avg_history[-1],
                "val/recon_loss": val_recon_avg_history[-1],
                # Other metrics
                "val/l1": val_l1_avg(None),
                "val/mse": val_mse_avg(None),
                "val/psnr": val_psnr_avg(None),
                "val/ssim": val_ssim_avg(None),
                "recon_comparison": wandb.Image(
                    comparison,
                    caption=f"Left: Original, Right: Reconstructed (Epoch {epoch})"
                )
            }, step=epoch)

        # Prepare checkpoint dict for best model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss_history': train_loss_avg_history,
            'train_recon_history': train_recon_avg_history,
            'val_loss_history': val_loss_avg_history,
            'val_recon_history': val_recon_avg_history,
        }
        # Save checkpoint periodico (ultimo stato) anche se scatta early stopping
        checkpoint_last_path = str(checkpoint_path).replace('.pt', '_last.pt') if checkpoint_path else None
        if checkpoint_last_path and (epoch % save_checkpoint_every == 0 or epoch == num_epochs):
            torch.save(checkpoint, checkpoint_last_path)
            print(f"Checkpoint saved at epoch {epoch}")
            wandb.save(checkpoint_last_path)

        # Early stopping step
        should_stop = es.step(val_loss_avg_history[-1], model=model, checkpoint=checkpoint)
        if should_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    return train_loss_avg_history, val_loss_avg_history


def step_vqvae(model, loss_fn, xb, device, opt=None):
    image_batch = xb.to(device)
    target_batch = image_batch  # For autoencoders, target is the input itself
    x_recon, vq_loss = model(image_batch)
    recon_loss = loss_fn(x_recon, target_batch)
    loss = recon_loss + vq_loss

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        opt.step()

    return loss.item(), recon_loss.item(), vq_loss.item(), x_recon.detach()


def fit_vqvae(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    num_epochs,
    device,
    start_epoch=1,
    checkpoint_path=None,
    train_loss_history=None,
    train_recon_history=None,
    val_loss_history=None,
    val_recon_history=None,
    save_checkpoint_every=1
):
    # Initialize loss histories
    train_loss_avg_history = train_loss_history if train_loss_history is not None else []
    train_recon_avg_history = train_recon_history if train_recon_history is not None else []
    val_loss_avg_history = val_loss_history if val_loss_history is not None else []
    val_recon_avg_history = val_recon_history if val_recon_history is not None else []

    fixed_batch = None  # To store a fixed batch for reconstruction visualization
    model = model.to(device)

    print(f"Training from epoch {start_epoch} to {num_epochs}...")

    # Path per best model (early stopping)
    checkpoint_best_path = str(checkpoint_path).replace('.pt', '_best.pt') if checkpoint_path else None
    es = EarlyStopping(patience=5, min_delta=0.0, mode='min', path=checkpoint_best_path)

    for epoch in range(start_epoch, num_epochs + 1):
        # --- TRAIN ---
        model.train()
        train_loss_avg = make_averager()
        train_recon_avg = make_averager()
        train_vq_avg = make_averager()

        batch_bar = tqdm(train_loader, leave=False, desc=f'epoch {epoch} train', total=len(train_loader))
        for image_batch, _ in batch_bar:
            loss, recon_loss, vq_loss, x_recon = step_vqvae(
                model, loss_fn, image_batch, device, optimizer
            )

            train_loss_avg(loss)
            train_recon_avg(recon_loss)
            train_vq_avg(vq_loss)
            refresh_bar(
                batch_bar, f"train batch [loss: {train_loss_avg(None):.3f} | recon loss: {train_recon_avg(None):.3f} | vq: {train_vq_avg(None):.3f}]")

        train_loss_avg_history.append(train_loss_avg(None))
        train_recon_avg_history.append(train_recon_avg(None))
        train_vq_avg = train_vq_avg(None)

        # --- VALIDATION ---
        model.eval()
        val_loss_averager = make_averager()
        val_recon_averager = make_averager()
        val_vq_averager = make_averager()

        val_l1_avg = make_averager()
        val_mse_avg = make_averager()
        val_psnr_avg = make_averager()
        val_ssim_avg = make_averager()

        with torch.no_grad():
            val_bar = tqdm(val_loader, total=len(val_loader), desc=f'epoch {epoch} validation')
            for image_batch, _ in val_bar:
                loss, recon_loss, vq_loss, x_recon = step_vqvae(
                    model, loss_fn, image_batch, device, opt=None
                )
                val_loss_averager(loss)
                val_recon_averager(recon_loss)
                val_vq_averager(vq_loss)

                metrics = compute_image_metrics(x_recon, image_batch.to(device))
                val_l1_avg(metrics['l1'])
                val_mse_avg(metrics['mse'])
                val_psnr_avg(metrics['psnr'])
                val_ssim_avg(metrics['ssim'])

                refresh_bar(
                    val_bar, f"validation batch [loss: {val_loss_averager(None):.3f} | recon loss: {val_recon_averager(None):.3f} | vq: {val_vq_averager(None):.3f}]")

            val_loss_avg_history.append(val_loss_averager(None))
            val_recon_avg_history.append(val_recon_averager(None))
            val_vq_avg = val_vq_averager(None)

        print(
            f"Epoch {epoch}: "
            f"Train Loss={train_loss_avg_history[-1]:.4f} (recon loss={train_recon_avg_history[-1]:.4f}, vq={train_vq_avg:.4f})\n"
            f"Val Loss={val_loss_avg_history[-1]:.4f} (recon loss={val_recon_avg_history[-1]:.4f}, vq={val_vq_avg:.4f})\n"
        )

        # logging wandb (and visualization)
        with torch.no_grad():
            if fixed_batch is None:
                sample_batch = next(iter(val_loader))
                fixed_batch = sample_batch[0][:4].clone() # first 4 images

            fixed_images = fixed_batch.to(device)
            fixed_recon, *_ = model(fixed_images)

            originals = fixed_images.cpu()
            reconstructions = fixed_recon.cpu()

            # 2x2 grid for originals
            orig_top = torch.cat([originals[0], originals[1]], dim=2)  # horizontal concat
            orig_bottom = torch.cat([originals[2], originals[3]], dim=2)
            grid_original = torch.cat([orig_top, orig_bottom], dim=1)  # vertical concat

            # 2x2 grid for reconstructions
            recon_top = torch.cat([reconstructions[0], reconstructions[1]], dim=2)
            recon_bottom = torch.cat([reconstructions[2], reconstructions[3]], dim=2)
            grid_recon = torch.cat([recon_top, recon_bottom], dim=1)

            # Concatenate original and reconstructed grids
            comparison = torch.cat([grid_original, grid_recon], dim=2)  # horizontal concat

            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss_avg_history[-1],
                "train/recon_loss": train_recon_avg_history[-1],
                "train/vq_loss": train_vq_avg,
                "val/loss": val_loss_avg_history[-1],
                "val/recon_loss": val_recon_avg_history[-1],
                "val/vq_loss": val_vq_avg,
                # other metrics
                "val/l1": val_l1_avg(None),
                "val/mse": val_mse_avg(None),
                "val/psnr": val_psnr_avg(None),
                "val/ssim": val_ssim_avg(None),
                "recon_comparison": wandb.Image(
                    comparison,
                    caption=f"Left: Original, Right: Reconstructed (Epoch {epoch})"
                )
            }, step=epoch)
        
        # Prepare checkpoint dict for best model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss_history': train_loss_avg_history,
            'train_recon_history': train_recon_avg_history,
            'val_loss_history': val_loss_avg_history,
            'val_recon_history': val_recon_avg_history,
        }
        # Save checkpoint periodico (ultimo stato) anche se scatta early stopping
        checkpoint_last_path = str(checkpoint_path).replace('.pt', '_last.pt') if checkpoint_path else None
        if checkpoint_last_path and (epoch % save_checkpoint_every == 0 or epoch == num_epochs):
            torch.save(checkpoint, checkpoint_last_path)
            print(f"Checkpoint saved at epoch {epoch}")
            wandb.save(checkpoint_last_path)
        # Early stopping step
        should_stop = es.step(val_loss_avg_history[-1], model=model, checkpoint=checkpoint)
        if should_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    return train_loss_avg_history, val_loss_avg_history


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
