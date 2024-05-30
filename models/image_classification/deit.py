from typing import Optional

import torch
from timm.models.deit import VisionTransformerDistilled
from timm.models import create_model

from models.image_classification.base import BaseTransformer
from utils.distillation_loss import DistillationLoss
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.args import get_args
from utils.load_data import get_train_test_loaders


class DeiT(BaseTransformer):
    def __init__(self, image_size: int, patch_size: int, num_layers: int, num_heads: int, hidden_dim: int,
                 mlp_ratio: int, dropout: int, attention_dropout: int, num_classes: int):
        self.img_size = image_size
        self.patch_size = patch_size
        self.depth = num_layers
        self.num_heads = num_heads
        self.embed_dim = hidden_dim
        self.mlp_ratio = mlp_ratio
        self.drop_rate = dropout
        self.attn_drop = attention_dropout
        self.num_classes = num_classes

        self.device = 'cuda'

    def _get_teacher_model(self, model_name='regnety_160'):
        return create_model(model_name, pretrained=True, num_classes=self.num_classes, global_pool='avg').to(
            self.device)

    def train_model_with_distillation(self, train_loader: DataLoader, test_loader: DataLoader, epochs: int,
                    val_loader: Optional[DataLoader] = None):

        model = VisionTransformerDistilled(img_size=self.img_size, patch_size=self.patch_size, depth=self.depth,
                                           num_heads=self.num_heads, embed_dim=self.embed_dim, mlp_ratio=self.mlp_ratio,
                                           drop_rate=self.drop_rate, attn_drop_rate=self.attn_drop,
                                           num_classes=self.num_classes).to(self.device)
        teacher_model = self._get_teacher_model().eval().to(self.device)

        model.set_distilled_training(True) # For distilled learning, set True otherwise its ViT

        base_criterion = nn.CrossEntropyLoss()
        criterion = DistillationLoss(base_criterion=base_criterion, teacher_model=teacher_model,
                                     distillation_type='hard', alpha=0.5, tau=5.0)

        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        train_losses, val_losses, test_losses = [], [], []
        train_accuracies, val_accuracies, test_accuracies = [], [], []

        for epoch in range(epochs):
            model.train()
            running_train_loss = 0.0
            correct_train, total_train = 0, 0
            train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
            for images, labels in train_loader_tqdm:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                student_teacher_outputs = model(images)
                loss = criterion(images, student_teacher_outputs, labels)
                loss.backward()
                optimizer.step()

                student_output, teacher_output = student_teacher_outputs  # Extracting only student outputs
                running_train_loss += loss.item() * images.size(0)
                _, predicted = torch.max(student_output, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

                train_loader_tqdm.set_postfix({
                    "Train Loss": running_train_loss / total_train,
                    "Train Acc": correct_train / total_train
                })

            epoch_train_loss = running_train_loss / len(train_loader.dataset)
            epoch_train_accuracy = correct_train / total_train
            train_losses.append(epoch_train_loss)
            train_accuracies.append(epoch_train_accuracy)

            # Validation phase
            if val_loader:
                model.eval()
                running_val_loss = 0.0
                correct_val, total_val = 0, 0

                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        outputs = model(images)
                        loss = base_criterion(outputs, labels)
                        running_val_loss += loss.item() * images.size(0)
                        _, predicted = torch.max(outputs, 1)
                        total_val += labels.size(0)
                        correct_val += (predicted == labels).sum().item()

                epoch_val_loss = running_val_loss / len(val_loader.dataset)
                epoch_val_accuracy = correct_val / total_val
                val_losses.append(epoch_val_loss)
                val_accuracies.append(epoch_val_accuracy)
            else:
                epoch_val_loss = "N/A"
                epoch_val_accuracy = "N/A"

            # Testing phase
            model.eval()
            running_test_loss = 0.0
            correct_test, total_test = 0, 0

            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = base_criterion(outputs, labels)
                    running_test_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total_test += labels.size(0)
                    correct_test += (predicted == labels).sum().item()

            epoch_test_loss = running_test_loss / len(test_loader.dataset)
            epoch_test_accuracy = correct_test / total_test
            test_losses.append(epoch_test_loss)
            test_accuracies.append(epoch_test_accuracy)

            tqdm.write(f"Epoch {epoch + 1}/{epochs} - "
                       f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.4f}, "
                       f"Val Loss: {epoch_val_loss}, Val Acc: {epoch_val_accuracy}, "
                       f"Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_accuracy:.4f}")

        return {"train_loss": train_losses, "val_loss": val_losses if val_loader else None, "test_loss": test_losses,
                "train_accuracy": train_accuracies, "val_accuracy": val_accuracies if val_loader else None,
                "test_accuracy": test_accuracies}


if __name__ == '__main__':

    args = get_args('deit_tinydistil_cifar100')
    # args = get_args('deit_tiny_cifar100')

    # Getting loaders
    train_loader, val_loader, test_loader = get_train_test_loaders(dataset_name="cifar100", batch_size=256,
                                                                   val_split=0.2, num_workers=4)

    deit = DeiT(image_size=args["image_size"], patch_size=args["patch_size"], num_layers=args["num_layers"],
                num_heads=args["num_heads"], hidden_dim=args["embed_dim"], mlp_ratio=args["mlp_ratio"],
                dropout=args["dropout"], attention_dropout=args["attention_dropout"],
                num_classes=args["num_classes"])

    # Training model
    if args['distilled_training']:
        # Epoch 100/100 - Train Loss: 2.1546, Train Acc: 0.8789,
        # Val Loss: 3.083437201309204, Val Acc: 0.2973,
        # Test Loss: 2.9748, Test Acc: 0.3269
        metrics = deit.train_model_with_distillation(train_loader=train_loader, val_loader=val_loader,
                                                     test_loader=test_loader, epochs=100)
    else:
        model = VisionTransformerDistilled(img_size=args["image_size"], patch_size=args["patch_size"],
                                           depth=args["num_layers"], num_heads=args["num_heads"],
                                           embed_dim=args["embed_dim"], mlp_ratio=args["mlp_ratio"],
                                           drop_rate=args["dropout"], attn_drop_rate=args["attention_dropout"],
                                           num_classes=args["num_classes"]).to('cuda')
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        deit.train_model(model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                         epochs=20, criterion=criterion, optimizer=optimizer)
