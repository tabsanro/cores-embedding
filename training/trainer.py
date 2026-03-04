"""
Training loop for both Baseline and CoRes models.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from training.losses import CoResLoss, BaselineLoss
from models.vcores import VCoResLoss


class Trainer:
    """Unified trainer for Baseline and CoRes models."""

    def __init__(self, model, config, model_type="cores"):
        """
        Args:
            model: BaselineModel or CoResModel instance.
            config: Configuration dict.
            model_type: "baseline" or "cores".
        """
        self.model = model
        self.config = config
        self.model_type = model_type
        self.device = torch.device(config["experiment"].get("device", "cuda"))

        # Move model to device
        self.model = self.model.to(self.device)

        # Multi-GPU support
        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            print(f"  Using {torch.cuda.device_count()} GPUs (DataParallel)")
            self.model = nn.DataParallel(self.model)
        self.raw_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        # Setup optimizer
        self._setup_optimizer()

        # Setup loss function
        self._setup_loss()

        # Setup output directory
        self.output_dir = os.path.join(
            config["experiment"]["output_dir"],
            config["experiment"]["name"],
            model_type,
        )
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(os.path.join(self.output_dir, "logs"))

        # Training state
        self.current_epoch = 0
        self.best_loss = float("inf")

    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        tc = self.config["training"]

        if tc["optimizer"] == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=tc["learning_rate"],
                weight_decay=tc["weight_decay"],
            )
        elif tc["optimizer"] == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=tc["learning_rate"],
                momentum=0.9,
                weight_decay=tc["weight_decay"],
            )
        else:
            raise ValueError(f"Unknown optimizer: {tc['optimizer']}")

        # Scheduler
        if tc["scheduler"] == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=tc["epochs"], eta_min=1e-6,
            )
        elif tc["scheduler"] == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1,
            )
        else:
            self.scheduler = None

    def _setup_loss(self):
        """Setup loss function based on model type."""
        tc = self.config["training"]

        if self.model_type == "vcores":
            vcores_tc = tc.get("vcores", {})
            self.criterion = VCoResLoss(
                concept_weight=tc.get("concept_loss_weight", 1.0),
                recon_weight=vcores_tc.get("recon_weight", 1.0),
                kl_weight=vcores_tc.get("kl_weight", 1.0),
                residual_reg_weight=tc.get("residual_reg_weight", 0.01),
                orthogonality_weight=tc.get("orthogonality_weight", 0.1),
                kl_annealing_epochs=vcores_tc.get("kl_annealing_epochs", 10),
            )
        elif self.model_type == "cores":
            self.criterion = CoResLoss(
                concept_weight=tc.get("concept_loss_weight", 1.0),
                residual_reg_weight=tc.get("residual_reg_weight", 0.01),
                orthogonality_weight=tc.get("orthogonality_weight", 0.1),
            )
        else:
            method = self.config["model"]["baseline"].get("method", "supervised")
            temperature = self.config["model"]["baseline"].get("temperature", 0.5)
            self.criterion = BaselineLoss(
                method=method, temperature=temperature,
            )

    def train_epoch(self, train_loader):
        """Train for one epoch.

        Args:
            train_loader: Training data loader.

        Returns:
            avg_loss: Average loss for the epoch.
            loss_dict: Dictionary of average losses.
        """
        self.model.train()
        total_losses = {}
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, (images, concept_labels, factor_labels) in enumerate(pbar):
            images = images.to(self.device)
            concept_labels = concept_labels.to(self.device)

            # Forward pass
            output = self.model(images)

            # Compute loss (V-CoRes needs original images and epoch for ELBO)
            if self.model_type == "vcores":
                loss, loss_dict = self.criterion(
                    output, concept_labels,
                    x_input=images, epoch=self.current_epoch,
                )
            else:
                loss, loss_dict = self.criterion(output, concept_labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Accumulate losses
            for key, val in loss_dict.items():
                total_losses[key] = total_losses.get(key, 0) + val
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        return avg_losses.get("loss_total", 0), avg_losses

    def validate(self, val_loader):
        """Validate model.

        Args:
            val_loader: Validation data loader.

        Returns:
            avg_loss: Average validation loss.
            metrics: Dictionary of validation metrics.
        """
        self.model.eval()
        total_losses = {}
        all_predictions = []
        all_labels = []
        num_batches = 0

        with torch.no_grad():
            for images, concept_labels, factor_labels in val_loader:
                images = images.to(self.device)
                concept_labels = concept_labels.to(self.device)

                output = self.model(images)
                if self.model_type == "vcores":
                    loss, loss_dict = self.criterion(
                        output, concept_labels,
                        x_input=images, epoch=self.current_epoch,
                    )
                else:
                    loss, loss_dict = self.criterion(output, concept_labels)

                # Accumulate
                for key, val in loss_dict.items():
                    total_losses[key] = total_losses.get(key, 0) + val
                num_batches += 1

                # Collect predictions for accuracy
                preds = (torch.sigmoid(output["logits"]) > 0.5).float()
                all_predictions.append(preds.cpu())
                all_labels.append(concept_labels.cpu())

        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}

        # Compute accuracy
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        accuracy = (all_predictions == all_labels).float().mean().item()

        metrics = {**avg_losses, "accuracy": accuracy}

        return avg_losses.get("loss_total", 0), metrics

    def train(self, train_loader, val_loader, epochs=None):
        """Full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Number of epochs (overrides config).

        Returns:
            history: Training history.
        """
        if epochs is None:
            epochs = self.config["training"]["epochs"]

        history = {"train": [], "val": []}
        save_every = self.config["training"].get("save_every", 10)
        eval_every = self.config["training"].get("eval_every", 5)

        print(f"\n{'='*60}")
        print(f"Training {self.model_type.upper()} model")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"{'='*60}\n")

        for epoch in range(epochs):
            self.current_epoch = epoch
            start_time = time.time()

            # Train
            train_loss, train_losses = self.train_epoch(train_loader)
            history["train"].append(train_losses)

            # Log to tensorboard
            for key, val in train_losses.items():
                self.writer.add_scalar(f"train/{key}", val, epoch)

            # Validate
            if epoch % eval_every == 0 or epoch == epochs - 1:
                val_loss, val_metrics = self.validate(val_loader)
                history["val"].append(val_metrics)

                for key, val in val_metrics.items():
                    self.writer.add_scalar(f"val/{key}", val, epoch)

                print(
                    f"Epoch {epoch:3d} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f} | "
                    f"Time: {time.time() - start_time:.1f}s"
                )

                # Save best model
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint("best.pt")

            # Save periodic checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(f"epoch_{epoch}.pt")

            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

        # Save final model
        self.save_checkpoint("final.pt")
        self.writer.close()

        return history

    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        path = os.path.join(self.output_dir, "checkpoints", filename)
        torch.save({
            "epoch": self.current_epoch,
            "model_state_dict": self.raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "config": self.config,
            "model_type": self.model_type,
        }, path)

    def load_checkpoint(self, filename):
        """Load model checkpoint."""
        path = os.path.join(self.output_dir, "checkpoints", filename)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.raw_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_loss = checkpoint["best_loss"]
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
