import torch
from tqdm import tqdm
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datasets import create_dataloader
from utils import (
    available_device,
    create_loss,
    create_optimizer,
)
from models import create_model


class Trainer:
    def __init__(
        self,
        dataset_name,
        model_name,
        arch_type,
        optimizer_name,
        loss_name,
        project_name,
        epochs=100,
        hidden=512,
    ):
        self.device = available_device()
        self.loss = create_loss(loss_name)
        num_classes = 10
        if dataset_name == "CIFAR100":
            num_classes = 100
        self.model = create_model(model_name, arch_type, num_classes=num_classes, hidden=hidden)
        self.model = self.model.to(device=self.device)
        self.optimizer = create_optimizer(optimizer_name, self.model.parameters(), weight_decay=0.0005)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer)
        self.train_dataloader, self.test_dataloader = create_dataloader(dataset_name)
        self.epochs = epochs
        self._wandb_logger = wandb.init(
            project=project_name,
        )

    def train(self):
        for epoch in range(self.epochs):
            running_loss = 0.0
            self.model.train()
            for i, data in tqdm(enumerate(self.train_dataloader, 0), desc=f"epoch = {epoch}/{self.epochs}"):
                inputs, labels = data[0].to(device=self.device), data[1].to(device=self.device)
                self.optimizer.zero_grad()

                outputs = self.model(inputs)

                loss = self.loss(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            self._wandb_logger.log(
                {
                    f"{epoch}": epoch,
                    f"train/losses": running_loss/len(self.train_dataloader),
                }
            )
            running_loss = self.evaluate()
            self.lr_scheduler.step(running_loss)

    def evaluate(self):
        correct = 0
        total = 0
        running_loss = 0
        with torch.no_grad():
            self.model.eval
            for data in self.test_dataloader:
                images, labels = data[0].to(device=self.device), data[1].to(device=self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = self.loss(outputs, labels)
                running_loss += loss.item()
            self._wandb_logger.log(
                {
                    f"validation/loss": running_loss/len(self.test_dataloader),
                    f"validation/accuracy": 100*correct/total,
                }
            )
            print(f"Accuracy of the network on validation set: %d %%" % (100*correct/total))
        return running_loss


