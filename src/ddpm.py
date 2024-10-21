import os
import torch
import torch.nn as nn

from src.utils import plot_loss, save_checkpoint


def train_model(
    model: nn.Module,
    optim: torch.optim.Optimizer,
    loss_fn: nn.Module,
    loaders: dict[str, torch.utils.data.DataLoader],
    beta_schedule: torch.Tensor,
    epochs: int = 10,
    valid_every: int = 1,
    T: int = 1000,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    using_diffusers: bool = False,
    checkpoint_dir: str = "checkpoints",
    checkpoint_every: int = 5,
) -> tuple[list[float], list[float]]:
    """
    Train the model for the specified number of epochs.

    Args:
        model (nn.Module): The model to train.
        optim (torch.optim.Optimizer): The optimizer to use.
        loss_fn (nn.Module): The loss function.
        loaders (dict[str, torch.utils.data.DataLoader]): DataLoader for training data and validation data.
        beta_schedule (torch.Tensor): The beta schedule for diffusion.
        epochs (int, optional): Number of epochs to train. Defaults to 10.
        valid_every (int, optional): Validate every n epochs. Defaults to 1.
        T (int, optional): Total number of diffusion steps. Defaults to 1000.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler. Defaults to None.
        using_diffusers (bool, optional): Whether using HuggingFace Diffusers library. Defaults to False.
        checkpoint_dir (str, optional): Directory to save checkpoints. Defaults to 'checkpoints'.
        checkpoint_every (int, optional): Save checkpoint every n epochs. Defaults to 5.

    Returns:
        tuple[list[float], list[float]]: The training and validation losses.
    """

    all_train_loss = []
    all_valid_loss = []

    alpha_bar = get_alpha_bar(beta_schedule)

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        print(f"Epoch {epoch} of {epochs}")
        model.train()
        train_loss = []

        for batch in loaders["train"]:
            x, _ = batch
            (noisy_images, t), e = prepare_batch(x, T, alpha_bar)

            if using_diffusers:
                y_pred = model(
                    noisy_images, t
                ).sample  # sample for diffusers library components
            else:
                y_pred = model(noisy_images, t)

            optim.zero_grad()
            loss = loss_fn(y_pred, e)
            loss.backward()
            optim.step()

            if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

        all_train_loss.append(sum(train_loss) / len(train_loss))

        if scheduler is not None and (
            not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)
        ):
            scheduler.step()

        if epoch % valid_every == 0:
            validate_model(
                model,
                loaders["valid"],
                loss_fn,
                all_valid_loss,
                T,
                alpha_bar,
                using_diffusers,
            )
            print(
                f"Epoch {epoch}, Train Loss: {sum(train_loss) / len(train_loss)}, "
                f"Valid Loss: {all_valid_loss[-1]}"
            )

        if epoch % checkpoint_every == 0:
            save_checkpoint(
                model, epoch, checkpoint_dir, optim, all_train_loss, all_valid_loss
            )

    plot_loss(all_train_loss, all_valid_loss)

    return all_train_loss, all_valid_loss


def prepare_batch(x: torch.Tensor, T: int, alpha_bar: torch.Tensor):
    """
    Prepare a batch for training by generating random timesteps and adding noise to the image.
    We use the random timesteps to generate the amount of noise that should be added to the image at that timestep.
    Then, we add the noise to the image.

    Args:
        x: the input image
        T: the number of diffusion steps
        beta_schedule: the beta values for each timestep

    Returns:
        the noisy image and the timestep
        the noise that was added to the image
    """
    t = torch.randint(0, T, (x.shape[0],), requires_grad=False).to(x.device)
    e = torch.randn_like(x, requires_grad=False).to(x.device)

    alpha_bar_t = alpha_bar[t].view(-1, 1, 1, 1)
    noisy_images = (alpha_bar_t.sqrt() * x) + ((1 - alpha_bar_t).sqrt() * e)

    return (noisy_images, t), e


def get_alpha_bar(beta_schedule: torch.Tensor):
    """
    what we need to do here is prepare a tensor of alpha_bars where each t'th entry
    in alphabars is the product of the alphas leading up to it. Alpha is  as 1 - beta
    """
    alpha = 1.0 - beta_schedule
    return alpha.cumprod(dim=0)


def validate_model(
    model: nn.Module,
    valid_loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    all_valid_loss: list,
    T: int,
    alpha_bar: torch.Tensor,
    using_diffusers: bool = False,
):
    model.eval()
    valid_loss = []

    with torch.no_grad():
        for batch in valid_loader:
            x, _ = batch
            (noisy_images, t), e = prepare_batch(x, T, alpha_bar=alpha_bar)

            if using_diffusers:
                y_pred = model(
                    noisy_images, t
                ).sample  # sample for diffusers library components
            else:
                y_pred = model(noisy_images, t)

            loss = loss_fn(y_pred, e)
            valid_loss.append(loss.item())

    avg_valid_loss = sum(valid_loss) / len(valid_loss)
    all_valid_loss.append(avg_valid_loss)


def sample_images(
    model,
    beta_schedule,
    T,
    device,
    num_samples=16,
    in_channels=1,
    using_diffusers: bool = False,
):
    model.eval()
    with torch.no_grad():
        x = torch.randn(num_samples, in_channels, 32, 32).to(device)
        alpha = 1.0 - beta_schedule
        alpha_bar = get_alpha_bar(beta_schedule)

        for t in reversed(range(T)):
            time_input = torch.full((num_samples,), t, device=device)
            beta_t = beta_schedule[t]
            sigma_t = beta_t.sqrt()

            alpha_t = alpha[t] if t > 0 else torch.tensor(1.0)
            alpha_bar_t = alpha_bar[t]

            alpha_bar_t = alpha_bar[t]
            z = (
                torch.randn_like(x).to(device)
                if t > 0
                else torch.zeros_like(x).to(device)
            )

            if using_diffusers:
                model_pred = model(x, time_input).sample
            else:
                model_pred = model(x, time_input)

            if t > 0:
                x_t_minus_1 = (1 / alpha_t.sqrt()) * (
                    x - (1 - alpha_t) / (1 - alpha_bar_t).sqrt() * model_pred
                ).clamp(-1, 1) + sigma_t * z
            else:
                x_t_minus_1 = (1 / alpha_t.sqrt()) * (
                    x - (1 - alpha_t) / (1 - alpha_bar_t).sqrt() * model_pred
                ).clamp(-1, 1)

            x = x_t_minus_1

    return x
