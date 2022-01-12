import torch
from tqdm import tqdm

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
        epochs=100,
    ):
        self.device = available_device()
        self.loss = create_loss(loss_name)
        self.model = create_model(model_name, arch_type)
        if dataset_name == "CIFAR100":
            self.model = create_model(model_name, arch_type)
        self.model = self.model.to(device=self.device)
        self.optimizer = create_optimizer(optimizer_name, self.model.parameters())

        self.train_dataloader, self.test_dataloader = create_dataloader(dataset_name)
        self.epochs = epochs

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
            self.evaluate()

    def evaluate(self):
        correct = 0
        total = 0
        with torch.no_grad():
            self.model.eval
            for data in self.test_dataloader:
                images, labels = data[0].to(device=self.device), data[1].to(device=self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print(f"Accuracy of the network on validation set: %d %%" % (100*correct/total))


