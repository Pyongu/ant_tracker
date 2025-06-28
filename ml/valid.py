import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
from antsdataset import get_coco_dataset
from torch.utils.data import DataLoader
from test import iou, nms, get_model
from collections import defaultdict

def calculate_mAP(det_dict, gt_dict, iou_threshold=0.5):
    aps = []

    gt_matched = {}
    total_true_bboxes = 0
    amount_bboxes = defaultdict(int)
    total_detections = 0
    for image_id in range(len(gt_dict)):
        amount_bboxes[image_id] = len(gt_dict[image_id])
        gt_matched[image_id] = [False] * len(gt_dict[image_id])
        total_true_bboxes += len(gt_dict[image_id])
        total_detections += len(det_dict[image_id][0])
    #print(amount_bboxes)
    
    # for key, val in amount_bboxes.items():
    #     amount_bboxes[key] = torch.zeros(val)
    
    #print(amount_bboxes)

    # Sorts by Prediction score
    det_dict = dict(sorted(det_dict.items(), key=lambda item : item[1][0][0][4], reverse = True))
    #print(len(det_dict))
    
    # gt_matched = [False for im_gts in gt_dict]

    # print("gt_matched: ", gt_matched)
    # print("total_true_boxes: ", total_true_bboxes)

    # Initization of TP and FP
    tp = [0] * total_detections
    fp = [0] * total_detections
    #print("tp: ", len(tp) , "fp: ", len(fp))

    global_det_idx = 0
    for image_idx in range(len(det_dict)):
        for det_idx, det_list in enumerate(det_dict[image_idx][0]):
            #print("det_idx: ", det_idx, "det_list[:4]: ", det_list[:4])
            #print("image_idex: ", image_idx, "det_idx: :", det_idx)
            #print("image number: ", image_idx)
            #print("length of det_list: ", len(det_list))
            best_iou = -1
            best_gt_idx = -1
            for gt_idx, gt in enumerate(gt_dict[image_idx]):
                #print("gt_idx", gt_idx,"gt: ", gt)
                iou_value = iou(det_list[:4], gt)

                if iou_value > best_iou:
                    best_iou = iou_value
                    #print("gt_idx: ", gt_idx)
                    best_gt_idx = gt_idx
            #print("best GT IDX:", best_gt_idx)
            #print("gt_matched: ", gt_matched[image_idx])
            if best_iou < iou_threshold or gt_matched[image_idx][best_gt_idx]:
                #print("det_idx: ", det_idx)
                fp[global_det_idx] = 1
            else:
                tp[global_det_idx] = 1
                gt_matched[image_idx][best_gt_idx] = True
            global_det_idx += 1

    #print("FP: ", FP)
    #print("TP: ", TP)
    # Cumulative tp and fp
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    eps = np.finfo(np.float32).eps
    recalls = tp / np.maximum(total_true_bboxes, eps)
    precisions = tp / np.maximum((tp + fp), eps)

    # Calculating area underneath the recall vs precisions curve
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    # area
    # for i in range(precisions.size - 1, 0, -1):
    #     precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
    # i = np.where(recalls[1:] != recalls[:-1])[0]
    # # Add the rectangular areas to get ap
    # ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])

    ap = 0.0
    for interp_pt in np.arange(0, 1 + 1E-3, 0.1):
        # Get precision values for recall values >= interp_pt
        prec_interp_pt = precisions[recalls >= interp_pt]
        
        # Get max of those precision values
        prec_interp_pt = prec_interp_pt.max() if prec_interp_pt.size > 0.0 else 0.0
        ap += prec_interp_pt
    ap = ap / 11.0
    aps.append(ap)
    #mean_ap =  sum(aps) / (len(aps) + 1E-6)
    return ap, tp, fp

def evaluate_mAP(data):
    # dictionary of lists that contains all of the bbox by image_id
    # x1, y1, x2, y2
    gt_boxes = defaultdict(list)
    det_boxes = defaultdict(list)
    # I don't like this implemtation change later
    image_idx = 0
    for images, targets in data:
        for target in targets:
            for obj in target:
                # Extract bbox
                bbox = obj["bbox"]  # Format: [x, y, width, height]
                x, y, w, h = bbox
                image_id = obj["image_id"]

                # Ensure the width and height are positive
                if w > 0 and h > 0:
                    gt_boxes[image_id].append([x, y, x + w, y + h])
        
        for image in images:
            with torch.no_grad():
                image = image.unsqueeze(0)
                image = image.to(device) 
                prediction = model(image)
            
            boxes = prediction[0]['boxes'].cpu().numpy()  # Get predicted bounding boxes
            scores = prediction[0]['scores'].cpu().numpy()  # Get predicted scores
            boxes_with_scores = np.hstack([boxes, scores[:, np.newaxis]])
            # Remove overlapping boxes
            nms_boxes = nms(boxes_with_scores, iou_threshold=0.3)
            det_boxes[image_idx].append(nms_boxes)
            image_idx += 1
        # print("gt_boxes: ", gt_boxes)
        print("det_boxes: ", det_boxes)
    
    #print(gt_boxes)
    #print(det_boxes)
    
    # ap = calculate_mAP(det_boxes, gt_boxes, iou_threshold=0.5)
    # print(ap)
    # return ap
    return 5

# Initialize the model
num_classes = 2 # Background + ant

# Move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# val_dataset = get_coco_dataset(
#     # img_dir="/Users/pk_3/My_Documents/AntProjectSM2025/ant_tracker-1/ml/ants.v2i.coco/valid",
#     # ann_file="/Users/pk_3/My_Documents/AntProjectSM2025/ant_tracker-1/ml/ants.v2i.coco/valid/_annotations.coco.json"
#     img_dir="/Users/pk_3/My_Documents/AntProjectSM2025/ant_tracker-1/ml/natural_substrate/test/images",
#     ann_file="/Users/pk_3/My_Documents/AntProjectSM2025/ant_tracker-1/ml/natural_substrate/test/images/annotations.coco.json"
# )

# val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))


# Load the trained model
model = get_model(num_classes)
model.load_state_dict(torch.load("trainedModels/fasterrcnn_resnet50_epoch_5.pth"))
model.to(device)
model.eval()  # Set the model to evaluation mode

# Not sure if this is needed
COCO_CLASSES = {0: "Background", 1: "Ant"}

# evaluate_mAP(val_loader)
