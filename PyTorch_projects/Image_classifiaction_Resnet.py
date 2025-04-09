import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score


# ImageNet normalization stats
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

class TransferLearningModel(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        
        # Load pre-trained ResNet
        self.backbone = models.resnet50(weights='IMAGENET1K_V2')
        
        # Replace final layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = accuracy_score(y.cpu(), preds.cpu())
        self.log('val_loss', loss)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        # Different learning rates for backbone vs head
        params = [
            {"params": self.backbone.parameters(), "lr": self.hparams.learning_rate/10},
            {"params": self.backbone.fc.parameters(), "lr": self.hparams.learning_rate}
        ]
        return optim.Adam(params)

class TransferLearningModel(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        
        # Load pre-trained ResNet
        self.backbone = models.resnet50(weights='IMAGENET1K_V2')
        
        # Replace final layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = accuracy_score(y.cpu(), preds.cpu())
        self.log('val_loss', loss)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        # Different learning rates for backbone vs head
        params = [
            {"params": self.backbone.parameters(), "lr": self.hparams.learning_rate/10},
            {"params": self.backbone.fc.parameters(), "lr": self.hparams.learning_rate}
        ]
        return optim.Adam(params)


class ImageDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Use Oxford Flowers 102 dataset
        self.train_data = datasets.Flowers102(
            root=self.data_dir,
            split='train',
            transform=train_transform,
            download=True
        )
        
        self.val_data = datasets.Flowers102(
            root=self.data_dir,
            split='val',
            transform=val_transform,
            download=True
        )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)


# Initialize components
dm = ImageDataModule(data_dir='./data', batch_size=64)
model = TransferLearningModel(num_classes=102)

# Configure trainer
trainer = pl.Trainer(
    max_epochs=10,
    accelerator='auto',
    devices=1 if torch.cuda.is_available() else None,
    callbacks=[
        pl.callbacks.ModelCheckpoint(monitor='val_acc', mode='max'),
        pl.callbacks.EarlyStopping(monitor='val_acc', patience=3, mode='max')
    ]
)

# Train with fine-tuning
trainer.fit(model, dm)

# Freeze all layers except final during initial training
for param in self.backbone.parameters():
    param.requires_grad = False


# Add class weights to loss
class_weights = compute_class_weights(train_data)
criterion = nn.CrossEntropyLoss(weight=class_weights)


# Unfreeze last 3 residual blocks
for block in list(self.backbone.children())[-3:]:
    for param in block.parameters():
        param.requires_grad = True


# Load best checkpoint
best_model = TransferLearningModel.load_from_checkpoint(
    checkpoint_path='best_checkpoint.ckpt'
)

# Test evaluation
trainer.test(best_model, datamodule=dm)

#Inference
# Single image inference
def predict(image_path):
    img = Image.open(image_path).convert('RGB')
    img = val_transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = best_model(img)
    return torch.softmax(outputs, dim=1)
