import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Load Faster R-CNN with ResNet-50 backbone
def get_model(num_classes):
    # Load pre-trained Faster R-CNN
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Intersection over Union (IoU) calculates the overlap between boxes and removes boxes that overlap too much
def iou(box1, box2):
    #print("box1: ", box1, "box2: ", box2)
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# Non-Maximum Suppression algorithm
# takes the list of boxes and applies the suppression based on the IoU threshold
def nms(boxes, iou_threshold=0.3):
    if len(boxes) == 0:
        return []
    
    boxes = boxes[boxes[:, 4].argsort()[::-1]]  # Sort by score
    selected_boxes = []

    while len(boxes) > 0:
        chosen_box = boxes[0]
        selected_boxes.append(chosen_box)
        
        remaining_boxes = []
        for box in boxes[1:]:
            if iou(chosen_box, box) < iou_threshold:
                remaining_boxes.append(box)
        boxes = np.array(remaining_boxes)
    
    return np.array(selected_boxes)

def prepare_image(image_path):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    image = Image.open(image_path).convert("RGB")  # Open image
    image_tensor = F.to_tensor(image).unsqueeze(0)  # Convert image to tensor and add batch dimension
    return image_tensor.to(device)

def get_class_name(class_id):
    # `prediction` contains:
    # - boxes: predicted bounding boxes
    # - labels: predicted class labels
    # - scores: predicted scores for each box (confidence level)
    COCO_CLASSES = {0: "Background", 1: "Ant"}
    return COCO_CLASSES.get(class_id, "Unknown")
    
# Draw bounding boxes with the correct class names and increase image size
def draw_boxes(image, prediction, fig_size=(10, 10)):
    boxes = prediction[0]['boxes'].cpu().numpy()  # Get predicted bounding boxes
    labels = prediction[0]['labels'].cpu().numpy()  # Get predicted labels
    scores = prediction[0]['scores'].cpu().numpy()  # Get predicted scores

    if len(boxes) == 0:
        print("No Detections")
        return

    boxes_with_scores = np.hstack([boxes, scores[:, np.newaxis]])

    nms_boxes = nms(boxes_with_scores, iou_threshold=0.3)

    filtered_boxes = nms_boxes[:, :4]
    
    # Set a threshold for showing boxes (e.g., score > 0.5)
    threshold = 0.5
    
    # Set up the figure size to control the image size
    plt.figure(figsize=fig_size)  # Adjust the figure size here

    for box, label, score in zip(filtered_boxes, labels, scores):
        if score > threshold:
            x_min, y_min, x_max, y_max = box
            class_name = get_class_name(label)  # Get the class name
            plt.imshow(image)  # Display the image
            plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                              linewidth=2, edgecolor='r', facecolor='none'))
            plt.text(x_min, y_min, f"{class_name} ({score:.2f})", color='r')
    
    plt.axis('off')  # Turn off axis
    plt.savefig("result/result.png")

if __name__ == "__main__":
    # Initialize the model
    num_classes = 2 # Background + ant

    # Move model to GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')

    # Load the unseen image
    image_path = "/home/paulkim/Documents/BeeLabSM2025/ml-ant_tracker/ant_tracker/ml/testIM/anttest2.JPG"
    image_tensor = prepare_image(image_path)

    # Load the trained model
    model = get_model(num_classes)
    model.load_state_dict(torch.load("trainedModels/fasterrcnn_resnet50_epoch_5.pth"))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    # Display the image with bounding boxes and correct labels
    with torch.no_grad():  # Disable gradient computation for inference
        prediction = model(image_tensor)

    draw_boxes(Image.open(image_path), prediction, fig_size=(12, 10))  # Example of increased size