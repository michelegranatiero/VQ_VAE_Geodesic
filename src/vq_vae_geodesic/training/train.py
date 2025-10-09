import torch
from tqdm import tqdm
import wandb
from vq_vae_geodesic.utils import make_averager, refresh_bar


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
