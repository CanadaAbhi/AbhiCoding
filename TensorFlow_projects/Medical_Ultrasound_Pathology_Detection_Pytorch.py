#pip install torch torchvision opencv-python matplotlib
#pip install pycocotools

#at bash shell
#git clone https://github.com/matterport/Mask_RCNN.git
#cd Mask_RCNN


#Convert Annotations to COCO Format
#If your dataset is not in COCO format, convert it using the following structure

import json

def create_coco_annotations(image_dir, annotations):
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "pathology"}]
    }
    annotation_id = 1

    for idx, (image_name, ann) in enumerate(annotations.items()):
        # Add image info
        coco_format["images"].append({
            "id": idx,
            "file_name": image_name,
            "height": ann["height"],
            "width": ann["width"]
        })

        # Add annotations (bounding boxes and masks)
        for bbox, mask in zip(ann["bboxes"], ann["masks"]):
            coco_format["annotations"].append({
                "id": annotation_id,
                "image_id": idx,
                "category_id": 1,
                "bbox": bbox,
                "segmentation": mask,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0
            })
            annotation_id += 1

    # Save to JSON
    with open("annotations.json", "w") as f:
        json.dump(coco_format, f)

#dataloader
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F

class UltrasoundDataset(CocoDetection):
    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

        # Convert image to tensor
        img = F.to_tensor(img)

        # Convert targets to PyTorch tensors
        boxes = torch.as_tensor([obj["bbox"] for obj in target], dtype=torch.float32)
        labels = torch.as_tensor([obj["category_id"] for obj in target], dtype=torch.int64)
        
        masks = [obj["segmentation"] for obj in target]
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        return img, {"boxes": boxes, "labels": labels, "masks": masks}

import torchvision

# Load pre-trained Mask R-CNN model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

# Modify the number of classes (background + pathology)
num_classes = 2  # Background + pathology
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Update mask predictor as well
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
    in_features_mask, hidden_layer, num_classes)

import torch.optim as optim

# Load dataset and DataLoader
train_dataset = UltrasoundDataset("data/images", "data/annotations.json")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

# Set up optimizer and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):  # Adjust epochs as needed
    model.train()
    epoch_loss = 0

    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()

    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

from torchvision.ops import box_iou

def evaluate_model(model, test_loader):
    model.eval()
    iou_scores = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                iou = box_iou(output["boxes"], target["boxes"]).mean().item()
                iou_scores.append(iou)

    print(f"Mean IoU: {sum(iou_scores) / len(iou_scores):.4f}")

from PIL import Image

def predict_image(image_path):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)[0]

    return output  # Contains boxes, labels, scores, and masks

# Example usage:
output = predict_image("data/images/sample.jpg")
print(output)
