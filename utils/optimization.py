import optuna
import torch
from torch import nn, optim

from models.image_classification.vanilla_vit import ViT
from utils.args import get_args
from utils.load_data import get_train_test_loaders



def objective(trial):
    num_layers = trial.suggest_int("num_layers", 6, 12)
    mlp_dim = trial.suggest_int("mlp_dim", 512, 2048, step=128)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    attention_dropout = trial.suggest_float("attention_dropout", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])

    args = get_args("vit_tiny_cifar100")
                      num_classes=args["num_classes"])

    criterion = nn.CrossEntropyLoss()

    # Select optimizer
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "SGD":
        momentum = trial.suggest_float("momentum", 0.5, 0.9)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    elif optimizer_name == "RMSprop":
        alpha = trial.suggest_float("alpha", 0.9, 0.99)
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=alpha)

    # Training and validation loops
    def train_epoch(model, train_loader):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return running_loss / len(train_loader.dataset), correct / total

    def validate_epoch(model, val_loader):
        model.eval()
        running_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to('cuda'), labels.to('cuda')
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return running_loss / len(val_loader.dataset), correct / total

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader)
        val_loss, val_acc = validate_epoch(model, val_loader)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss


study = optuna.create_study(direction="minimize")

best_trial = study.best_trial

print(f"Best trial: {best_trial.number}")
print(f"  Value: {best_trial.value}")
print("  Params: ")
for key, value in best_trial.params.items():
    print(f"{key}: {value}")

# study = joblib.load('study.pkl')
