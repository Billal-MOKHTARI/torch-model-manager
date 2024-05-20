# from groundingdino.util.inference import load_model, load_image, predict, annotate
# import cv2
# import sys
# import os
# from segment_anything import SamPredictor, build_sam
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from diffusers import StableDiffusionInpaintPipeline
# from env import DEVICE
# import torch

# class SegmentationManager:
#     def __init__(self, sam_backbone = "sam_vit_h", g_dino_backbone = "swin_t", device = DEVICE):
#         g_dino_model_config_path = "GroundingDINO/groundingdino/config"
#         g_dino_checkpoint_path = "weights/GroundingDINO"
#         sam_checkpoint_path = "weights/SAM"
#         self.device = device

#         if sam_backbone == "sam_vit_h":
#             sam_checkpoint_path = os.path.join(sam_checkpoint_path, "sam_vit_h_4b8939.pth")
#             if not os.path.exists(sam_checkpoint_path):
#                 os.system(f"wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
#                 os.system(f"mv sam_vit_h_4b8939.pth {sam_checkpoint_path}")
                
#         elif sam_backbone == "sam_vit_b":
#             sam_checkpoint_path = os.path.join(sam_checkpoint_path, "sam_vit_b_01ec64.pth")
#             if not os.path.exists(sam_checkpoint_path):
#                 os.system(f"wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
#                 os.system(f"mv sam_vit_b_01ec64.pth {sam_checkpoint_path}")
                
#         elif sam_backbone == "sam_vit_l":
#             sam_checkpoint_path = os.path.join(sam_checkpoint_path, "sam_vit_l_0b3195.pth")
#             if not os.path.exists(sam_checkpoint_path):
#                 os.system(f"wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth")
#                 os.system(f"mv sam_vit_l_0b3195.pth {sam_checkpoint_path}")

#         if g_dino_backbone == "swin_t":
#             g_dino_checkpoint_path = os.path.join(g_dino_checkpoint_path, "groundingdino_swint_ogc.pth")
#             g_dino_model_config_path = os.path.join(g_dino_model_config_path, "GroundingDINO_SwinT_OGC.py")
            
#             if not os.path.exists(g_dino_checkpoint_path):
#                 os.system(f"wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth")
#                 os.system(f"mv groundingdino_swint_ogc.pth {g_dino_checkpoint_path}")

#         # Define DINO model
#         self.ground_dino = load_model(g_dino_model_config_path, g_dino_checkpoint_path, device = self.device)

#         # Define SAM model
#         self.sam = build_sam(checkpoint=sam_checkpoint_path)
#         self.sam.to(device=DEVICE)
#         # Define SAM predictor
#         self.sam_predictor = SamPredictor(self.sam)

#         # if DEVICE == 'cpu':
#         #     float_type = torch.float32
#         # else:
#         #     float_type = torch.float16

#         # self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
#         #     "stabilityai/stable-diffusion-2-inpainting",
#         #     torch_dtype=float_type,
#         # )

#         # if DEVICE.type != 'cpu':
#         #     self.pipe = self.pipe.to("cuda")

#     def detect_objects_image(self, image_path, text_prompt, box_threshold = 0.35, text_threshold = 0.25, output_path = None):
#         image_source, image = load_image(image_path)
#         boxes, logits, phrases = predict(
#         model = self.ground_dino,
#         image = image,
#         caption = text_prompt,
#         box_threshold = box_threshold,
#         text_threshold = text_threshold,
#         device = self.device
#         )
#         annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        
#         if output_path is not None:
#             cv2.imwrite(output_path, annotated_frame)

#         return boxes, logits, phrases, annotated_frame


# seg_m = SegmentationManager()
# boxes, logits, phrases, _ = seg_m.detect_objects_image("cat.jpg", "cat", output_path = "cat_annotated.jpg")

# print(boxes)
# print(logits)
# print(phrases)


















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

        # Set SAM checkpoint path based on the backbone
        self.sam_checkpoint_path = os.path.join(sam_checkpoint_path, f"{sam_backbone}.pth")
        if not os.path.exists(self.sam_checkpoint_path):
            os.system(f"wget https://dl.fbaipublicfiles.com/segment_anything/{sam_backbone}.pth")
            os.system(f"mv {sam_backbone}.pth {self.sam_checkpoint_path}")

        # Set GroundingDINO checkpoint path and model config path
        self.g_dino_checkpoint_path = os.path.join(g_dino_checkpoint_path, "groundingdino_swint_ogc.pth")
        self.g_dino_model_config_path = os.path.join(g_dino_model_config_path, "GroundingDINO_SwinT_OGC.py")
        if not os.path.exists(self.g_dino_checkpoint_path):
            os.system(f"wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth")
            os.system(f"mv groundingdino_swint_ogc.pth {self.g_dino_checkpoint_path}")

        # Initialize models
        self.grounding_dino_model = Model(model_config_path=self.g_dino_model_config_path, 
                                          model_checkpoint_path=self.g_dino_checkpoint_path,
                                          device = self.device)
        sam = sam_model_registry[sam_backbone](checkpoint=self.sam_checkpoint_path)
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
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy),
            torch.from_numpy(detections.confidence),
            nms_threshold
        ).numpy().tolist()
        
        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]
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
        labels = [
            f"{classes[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _, _ 
            in detections
        ]
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        return annotated_image

    def save_image(self, image, path):
        cv2.imwrite(path, image)
        
    def grounding_sam(self, source_image_path, classes, box_threshold=0.25, text_threshold=0.25, nms_threshold=0.8):
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

        return annotated_image, detections

# Example usage
# if __name__ == "__main__":
#     SOURCE_IMAGE_PATH = "./assets/demo2.jpg"
#     CLASSES = ["The running dog"]
#     BOX_THRESHOLD = 0.25
#     TEXT_THRESHOLD = 0.25
#     NMS_THRESHOLD = 0.8

#     manager = SegmentationManager()

#     # Apply the grounding SAM pipeline
#     annotated_image, detections = manager.grounding_sam(SOURCE_IMAGE_PATH, CLASSES, BOX_THRESHOLD, TEXT_THRESHOLD, NMS_THRESHOLD)

#     # Save the final annotated image
#     manager.save_image(annotated_image, "grounded_sam_annotated_image.jpg")

    # If needed, further processing on detections can be done here