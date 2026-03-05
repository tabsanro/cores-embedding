"""Training loop for Baseline and CoRes models."""

import os
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from training.losses import CoResLoss, BaselineLoss
from models.vcores import VCoResLoss


class Trainer:
    """Unified trainer for Baseline and CoRes models."""

    def __init__(self, model, config, model_type="cores"):
        self.model = model.to(torch.device(config["experiment"].get("device", "cuda")))
        self.config = config
        self.model_type = model_type
        self.device = next(self.model.parameters()).device

        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            print(f"  Using {torch.cuda.device_count()} GPUs (DataParallel)")
            self.model = nn.DataParallel(self.model)
        self.raw_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        self._setup_optimizer()
        self._setup_loss()

        self.output_dir = os.path.join(
            config["experiment"]["output_dir"],
            config["experiment"]["name"],
            model_type,
        )
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        self.writer = SummaryWriter(os.path.join(self.output_dir, "logs"))
        self.current_epoch = 0
        self.best_loss = float("inf")

    def _setup_optimizer(self):
        tc = self.config["training"]
        opt_kwargs = dict(params=self.model.parameters(), lr=tc["learning_rate"], weight_decay=tc["weight_decay"])

        if tc["optimizer"] == "adam":
            self.optimizer = optim.Adam(**opt_kwargs)
        elif tc["optimizer"] == "sgd":
            self.optimizer = optim.SGD(**opt_kwargs, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {tc['optimizer']}")

        if tc["scheduler"] == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=tc["epochs"], eta_min=1e-6)
        elif tc["scheduler"] == "step":
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        else:
            self.scheduler = None

    def _setup_loss(self):
        tc = self.config["training"]
        if self.model_type == "vcores":
            vt = tc.get("vcores", {})
            self.criterion = VCoResLoss(
                concept_weight=tc.get("concept_loss_weight", 1.0),
                recon_weight=vt.get("recon_weight", 1.0),
                kl_weight=vt.get("kl_weight", 1.0),
                residual_reg_weight=tc.get("residual_reg_weight", 0.01),
                orthogonality_weight=tc.get("orthogonality_weight", 0.1),
                kl_annealing_epochs=vt.get("kl_annealing_epochs", 10),
            )
        elif self.model_type == "cores":
            self.criterion = CoResLoss(
                concept_weight=tc.get("concept_loss_weight", 1.0),
                residual_reg_weight=tc.get("residual_reg_weight", 0.01),
                orthogonality_weight=tc.get("orthogonality_weight", 0.1),
            )
        else:
            bm = self.config["model"]["baseline"]
            self.criterion = BaselineLoss(method=bm.get("method", "supervised"), temperature=bm.get("temperature", 0.5))

    def _compute_loss(self, output, concept_labels, images):
        if self.model_type == "vcores":
            return self.criterion(output, concept_labels, x_input=images, epoch=self.current_epoch)
        return self.criterion(output, concept_labels)

    def _run_epoch(self, loader, train=True):
        self.model.train(train)
        total_losses = defaultdict(float)
        all_preds, all_labels = [], []

        ctx = torch.enable_grad() if train else torch.no_grad()
        pbar = tqdm(loader, desc=f"Epoch {self.current_epoch}" if train else "Validation")

        with ctx:
            for images, concept_labels, _ in pbar:
                images = images.to(self.device)
                concept_labels = concept_labels.to(self.device)

                output = self.model(images)
                loss, loss_dict = self._compute_loss(output, concept_labels, images)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                else:
                    preds = (torch.sigmoid(output["logits"]) > 0.5).float()
                    all_preds.append(preds.cpu())
                    all_labels.append(concept_labels.cpu())

                for k, v in loss_dict.items():
                    total_losses[k] += v

        n = len(loader)
        avg_losses = {k: v / n for k, v in total_losses.items()}

        if not train:
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            avg_losses["accuracy"] = (all_preds == all_labels).float().mean().item()

        return avg_losses.get("loss_total", 0), avg_losses

    def train_epoch(self, train_loader):
        return self._run_epoch(train_loader, train=True)

    def validate(self, val_loader):
        return self._run_epoch(val_loader, train=False)

    def train(self, train_loader, val_loader, epochs=None):
        if epochs is None:
            epochs = self.config["training"]["epochs"]

        tc = self.config["training"]
        save_every = tc.get("save_every", 10)
        eval_every = tc.get("eval_every", 5)
        history = {"train": [], "val": []}

        print(f"\n{'='*60}\nTraining {self.model_type.upper()} | Device: {self.device} | Epochs: {epochs}\n{'='*60}\n")

        for epoch in range(epochs):
            self.current_epoch = epoch
            start_time = time.time()

            train_loss, train_losses = self.train_epoch(train_loader)
            history["train"].append(train_losses)
            for k, v in train_losses.items():
                self.writer.add_scalar(f"train/{k}", v, epoch)

            if epoch % eval_every == 0 or epoch == epochs - 1:
                val_loss, val_metrics = self.validate(val_loader)
                history["val"].append(val_metrics)
                for k, v in val_metrics.items():
                    self.writer.add_scalar(f"val/{k}", v, epoch)

                print(
                    f"Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                    f"Acc: {val_metrics['accuracy']:.4f} | Time: {time.time() - start_time:.1f}s"
                )

                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint("best.pt")

            if epoch % save_every == 0:
                self.save_checkpoint(f"epoch_{epoch}.pt")

            if self.scheduler is not None:
                self.scheduler.step()

        self.save_checkpoint("final.pt")
        self.writer.close()
        return history

    def save_checkpoint(self, filename):
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
        path = os.path.join(self.output_dir, "checkpoints", filename)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.raw_model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.current_epoch = ckpt["epoch"]
        self.best_loss = ckpt["best_loss"]
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
