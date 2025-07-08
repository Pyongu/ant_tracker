import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
from antsdataset import get_coco_dataset

# Load Faster R-CNN with ResNet-50 backbone
def get_model(num_classes):
    # Load pre-trained Faster R-CNN
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    for images, targets in data_loader:
        # Move images to the device
        images = [img.to(device) for img in images]

        # Validate and process targets
        processed_targets = []
        valid_images = []
        for i, target in enumerate(targets):
            boxes = []
            labels = []
            for obj in target:
                # Extract bbox
                bbox = obj["bbox"]  # Format: [x, y, width, height]
                x, y, w, h = bbox

                # Ensure the width and height are positive
                if w > 0 and h > 0:
                    boxes.append([x, y, x + w, y + h])  # Convert to [x_min, y_min, x_max, y_max]
                    labels.append(obj["category_id"])

            # Only process if there are valid boxes
            if boxes:
                processed_target = {
                    "boxes": torch.tensor(boxes, dtype=torch.float32).to(device),
                    "labels": torch.tensor(labels, dtype=torch.int64).to(device),
                }
                processed_targets.append(processed_target)
                valid_images.append(images[i])  # Add only valid images

        # Skip iteration if no valid targets
        # The issue is here
        if not processed_targets:
            continue
        
        # Ensure images and targets are aligned
        images = valid_images

        # Forward pass
        loss_dict = model(images, processed_targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backpropagation
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"Epoch [{epoch}] Loss: {losses.item():.4f}")


if __name__ == "__main__":

    # Load datasets
    train_dataset = get_coco_dataset(
    # img_dir="/Users/pk_3/My_Documents/AntProjectSM2025/ant_tracker-1/ml/ants.v2i.coco/train",
    # ann_file="/Users/pk_3/My_Documents/AntProjectSM2025/ant_tracker-1/ml/ants.v2i.coco/train/_annotations.coco.json"
    # img_dir="/Users/pk_3/My_Documents/AntProjectSM2025/ant_tracker-1/ml/natural_substrate/train/images",
    # ann_file="/Users/pk_3/My_Documents/AntProjectSM2025/ant_tracker-1/ml/natural_substrate/train/images/annotations.coco.json"
    # img_dir="/Users/pk_3/My_Documents/AntProjectSM2025/ant_tracker-1/ml/project-1-at-2025-07-02-17-29-a8cd87b5/images",
    # ann_file="/Users/pk_3/My_Documents/AntProjectSM2025/ant_tracker-1/ml/project-1-at-2025-07-02-17-29-a8cd87b5/result.json"
    # img_dir="/home/paulkim/Documents/BeeLabSM2025/ml-ant_tracker/ant_tracker/ml/Ant_dataset/OutdoorDataset/Seq0006Object21Image64/img",
    # ann_file="/home/paulkim/Documents/BeeLabSM2025/ml-ant_tracker/ant_tracker/ml/Ant_dataset/OutdoorDataset/Seq0006Object21Image64/annotations.coco.json"
    img_dir="/home/paulkim/Documents/BeeLabSM2025/ml-ant_tracker/ant_tracker/ml/ants.v2i.coco-20250708T213721Z-1-001/ants.v2i.coco/train",
    ann_file="/home/paulkim/Documents/BeeLabSM2025/ml-ant_tracker/ant_tracker/ml/ants.v2i.coco-20250708T213721Z-1-001/ants.v2i.coco/train/_annotations.coco.json"
    )

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # Initialize the model
    num_classes = 2 # Background + ant
    model = get_model(num_classes)
    # model.load_state_dict(torch.load("trainedModels/fasterrcnn_resnet50_epoch_10.pth"))

    # Move model to GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    model.to(device)

    # Define optimizer and learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        lr_scheduler.step()
        
        # Save the model's state dictionary after every epoch
        model_path = f"trainedModels/fasterrcnn_resnet50_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved: {model_path}")

