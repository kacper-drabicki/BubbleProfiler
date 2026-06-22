import torch


def train(model, loss_fn, optimizer, epochs, scheduler, config):
    # Select computation device (CPU or GPU) from config
    device = torch.device(config.device)

    # Set model to training mode (enables dropout, etc. if present)
    model.train()

    # Radial coordinate grid for solving the PDE
    # NOTE: start at 0.01 to avoid r=0 singularity in radial equations
    r = torch.linspace(0.01, config.r_max, 500, device=device).view(-1, 1).requires_grad_(True)

    # Instantiate optimizer and learning-rate scheduler from config
    optimizer = optimizer(model.parameters())
    scheduler = scheduler(optimizer)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        y = model(r)

        # Compute physics-informed loss:
        # includes PDE residual + boundary conditions
        loss = loss_fn(r, y)

        # Backpropagation through physics loss
        loss.backward()

        optimizer.step()

        # Update scheduler
        scheduler.step(loss.item())

        # Logging for monitoring convergence
        if epoch % 1000 == 0:
            print(f'Epoch: {epoch}; Loss: {loss.item():.10e}')


def pretrain(model, config):
    # runs training with simplfied boundary conditions to enfore general structure of the solution
    return train(
        model,
        loss_fn=config.pretrain_loss_fn,
        optimizer=config.pretrain_optimizer,
        epochs=config.pretrain_epochs,
        scheduler=config.pretrain_scheduler,
        config=config
    )


def finetune(model, config):
    # runs training using the correct boundary conditions
    return train(
        model,
        loss_fn=config.finetune_loss_fn,
        optimizer=config.finetune_optimizer,
        epochs=config.finetune_epochs,
        scheduler=config.finetune_scheduler,
        config=config
    )