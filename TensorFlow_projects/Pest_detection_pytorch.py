import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt


class PestDataset(Dataset):
    def __init__(self, img_dir, annotation_dir, transform=None):
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        ann_path = os.path.join(self.annotation_dir, self.img_names[idx].replace('.jpg', '.txt'))
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Parse annotations (class_id, x_center, y_center, width, height)
        boxes = []
        labels = []
        with open(ann_path) as f:
            for line in f.readlines():
                class_id, x, y, w, h = map(float, line.strip().split())
                x_min = (x - w/2) * image.shape[1]
                y_min = (y - h/2) * image.shape[0]
                x_max = (x + w/2) * image.shape[1]
                y_max = (y + h/2) * image.shape[0]
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(int(class_id))
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        return image, {"boxes": boxes, "labels": labels}

    def __len__(self):
        return len(self.img_names)


class PestDetectionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Backbone with attention (ResNet50 + FPN)
        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.fpn = torchvision.ops.FeaturePyramidNetwork([256, 512, 1024, 2048], 256)
        
        # Multi-task heads
        self.detection_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes + 4, kernel_size=1)
        )
        
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 1, kernel_size=1)
        )

    def forward(self, x):
        features = self.backbone(x)
        pyramid_features = self.fpn(features)
        
        # Detection output
        detections = self.detection_head(pyramid_features['p3'])
        
        # Segmentation output
        segmentation = self.segmentation_head(pyramid_features['p2'])
        
        return detections, segmentation


# Hyperparameters
num_classes = 5  # Ants, grasshoppers, etc.
batch_size = 8
lr = 0.001

# Dataset and DataLoader
train_dataset = PestDataset(img_dir='pests/train', annotation_dir='pests/train_labels')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model and optimizer
model = PestDetectionModel(num_classes=num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Loss functions
detection_criterion = nn.CrossEntropyLoss()
regression_criterion = nn.SmoothL1Loss()
segmentation_criterion = nn.BCEWithLogitsLoss()


# Hyperparameters
num_classes = 5  # Ants, grasshoppers, etc.
batch_size = 8
lr = 0.001

# Dataset and DataLoader
train_dataset = PestDataset(img_dir='pests/train', annotation_dir='pests/train_labels')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model and optimizer
model = PestDetectionModel(num_classes=num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Loss functions
detection_criterion = nn.CrossEntropyLoss()
regression_criterion = nn.SmoothL1Loss()
segmentation_criterion = nn.BCEWithLogitsLoss()


for epoch in range(10):
    model.train()
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        detections, segmentation = model(images)
        
        # Detection loss
        cls_logits = detections[:, :num_classes]
        box_regression = detections[:, num_classes:]
        cls_loss = detection_criterion(cls_logits, targets['labels'])
        reg_loss = regression_criterion(box_regression, targets['boxes'])
        
        # Segmentation loss
        seg_loss = segmentation_criterion(segmentation, targets['masks'])
        
        # Total loss
        loss = cls_loss + reg_loss + seg_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


def detect_pests(image_path, model):
    # Preprocess
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = F.to_tensor(image).unsqueeze(0).to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        detections, _ = model(tensor)
    
    # Post-process detections
    boxes = detections[0]['boxes'].cpu().numpy()
    labels = detections[0]['labels'].cpu().numpy()
    scores = detections[0]['scores'].cpu().numpy()
    
    # Visualize
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(image, f"Pest {label}: {score:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
    
    cv2.imshow("Pest Detection", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


