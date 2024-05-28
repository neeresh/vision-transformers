import optuna
import torch
from torch import nn, optim

from models.image_classification.vanilla_vit import ViT
from utils.args import get_args
from utils.load_data import get_train_test_loaders
import joblib

train_loader, val_loader, test_loader = get_train_test_loaders(dataset_name="cifar100", batch_size=256, val_split=0.2, num_workers=8)


def objective(trial):
    num_layers = trial.suggest_int("num_layers", 6, 12)
    num_heads = trial.suggest_int("num_heads", 6, 6)
    hidden_dim = trial.suggest_categorical("hidden_dim", [384, 576, 768, 960])
    mlp_dim = trial.suggest_int("mlp_dim", 512, 2048, step=128)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    attention_dropout = trial.suggest_float("attention_dropout", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])

    # Model
    args = get_args("vit_tiny_cifar100")
    model = ViT(image_size=args["image_size"], patch_size=args["patch_size"], num_layers=num_layers,
                      num_heads=num_heads, hidden_dim=hidden_dim, mlp_dim=mlp_dim,
                      dropout=dropout, attention_dropout=attention_dropout,
                      num_classes=args["num_classes"])
    model.to('cuda')

    # Criterion
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

    num_epochs = 50
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader)
        val_loss, val_acc = validate_epoch(model, val_loader)
        trial.report(val_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss


# Creating study and optimizing the objective function
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# Best trail
best_trial = study.best_trial

print(f"Best trial: {best_trial.number}")
print(f"Value: {best_trial.value}")
print("Params: ")
for key, value in best_trial.params.items():
    print(f"{key}: {value}")

# Saving the studey
joblib.dump(study, 'vanilla_vit.pkl')
# study = joblib.load('study.pkl')


# # Helper Methods
# def find_divisible_pairs(num_heads, hidden_dim_range):
#     divisible_pairs = []
#     for hidden_dim in hidden_dim_range:
#         if hidden_dim % num_heads == 0:
#             divisible_pairs.append((hidden_dim, num_heads))
#     return divisible_pairs
#
# num_heads = 6
# hidden_dim_range = range(256, 2048, 64)
# divisible_pairs = find_divisible_pairs(num_heads, hidden_dim_range)
#
# print("Pairs of (hidden_dim, num_heads) divisible by each other:")
# for pair in divisible_pairs:
#     print(pair)
