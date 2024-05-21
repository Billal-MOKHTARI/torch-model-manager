import os
import cv2
import numpy as np
import torch
import torchvision
import supervision as sv

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SegmentationManager:
    def __init__(self, sam_backbone="sam_vit_h", g_dino_backbone="swin_t", device=DEVICE):
        self.device = device
        
        # Paths for checkpoints and configs
        g_dino_model_config_path = "GroundingDINO/groundingdino/config"
        g_dino_checkpoint_path = "weights/GroundingDINO"
        sam_checkpoint_path = "weights/SAM"
        sam_type = None



        if sam_backbone == "sam_vit_h":
            sam_checkpoint_path = os.path.join(sam_checkpoint_path, "sam_vit_h_4b8939.pth")
            sam_type = "vit_h"
            if not os.path.exists(sam_checkpoint_path):
                os.system(f"wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
                os.system(f"mv sam_vit_h_4b8939.pth {sam_checkpoint_path}")
                
        elif sam_backbone == "sam_vit_b":
            sam_checkpoint_path = os.path.join(sam_checkpoint_path, "sam_vit_b_01ec64.pth")
            sam_type = "vit_b"

            if not os.path.exists(sam_checkpoint_path):
                os.system(f"wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
                os.system(f"mv sam_vit_b_01ec64.pth {sam_checkpoint_path}")
                
        elif sam_backbone == "sam_vit_l":
            sam_checkpoint_path = os.path.join(sam_checkpoint_path, "sam_vit_l_0b3195.pth")
            sam_type = "vit_l"

            if not os.path.exists(sam_checkpoint_path):
                os.system(f"wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth")
                os.system(f"mv sam_vit_l_0b3195.pth {sam_checkpoint_path}")

        if g_dino_backbone == "swin_t":
            g_dino_checkpoint_path = os.path.join(g_dino_checkpoint_path, "groundingdino_swint_ogc.pth")
            g_dino_model_config_path = os.path.join(g_dino_model_config_path, "GroundingDINO_SwinT_OGC.py")
            
            if not os.path.exists(g_dino_checkpoint_path):
                os.system(f"wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth")
                os.system(f"mv groundingdino_swint_ogc.pth {g_dino_checkpoint_path}")

        # Initialize models
        self.grounding_dino_model = Model(model_config_path=g_dino_model_config_path, 
                                          model_checkpoint_path=g_dino_checkpoint_path,
                                          device = self.device)

        sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint_path)
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)

    def load_image(self, image_path):
        return cv2.imread(image_path)

    def detect_objects(self, image, classes, box_threshold=0.25, text_threshold=0.25):
        detections = self.grounding_dino_model.predict_with_classes(
            image=image,
            classes=classes,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        return detections

    
    def apply_nms(self, detections, nms_threshold=0.8):
        def first_occurrences_indices(lst):
            seen = {}
            indices = []
            
            for i, value in enumerate(lst):
                if value not in seen:
                    seen[value] = i
                    indices.append(i)
            
            return indices
        
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy),
            torch.from_numpy(detections.confidence),
            nms_threshold
        ).numpy().tolist()
        
        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]
        idx = first_occurrences_indices(detections.class_id)

        detections.xyxy = detections.xyxy[idx]
        detections.confidence = detections.confidence[idx]
        detections.class_id = detections.class_id[idx]

        return detections
    
    def segment(self, image, xyxy):
        self.sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = self.sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    def annotate_image(self, image, detections, classes):
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()

        confidences = detections.confidence
        class_ids = detections.class_id
        labels = [f"{classes[class_id]} {confidence:0.2f}" for class_id, confidence in zip(class_ids, confidences)]
        print(labels)
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        return annotated_image

    def grounding_sam(self, source_image_path, classes, box_threshold=0.25, text_threshold=0.25, nms_threshold=0.8, output_path=None):
        # Load image
        image = self.load_image(source_image_path)

        # Detect objects
        detections = self.detect_objects(image, classes, box_threshold, text_threshold)

        # Apply NMS
        detections = self.apply_nms(detections, nms_threshold)

        # Convert detections to masks
        detections.mask = self.segment(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections.xyxy)

        # Annotate image with segmented masks
        annotated_image = self.annotate_image(image, detections, classes)

        if output_path is not None:
            self.save_image(annotated_image, output_path)

        return detections
    
    def save_image(self, image, path):
        cv2.imwrite(path, image)
# Example usage
if __name__ == "__main__":
    SOURCE_IMAGE_PATH = "cat.jpg"
    CLASSES = ["cat", "table", "window"]
    BOX_THRESHOLD = 0.25
    TEXT_THRESHOLD = 0.25
    NMS_THRESHOLD = 0.8

    manager = SegmentationManager()

    # Apply the grounding SAM pipeline
    annotated_image, detections = manager.grounding_sam(SOURCE_IMAGE_PATH, CLASSES, BOX_THRESHOLD, TEXT_THRESHOLD, NMS_THRESHOLD, "grounded_sam_annotated_image.jpg")