from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import sys
import os
from segment_anything import SamPredictor, build_sam
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from diffusers import StableDiffusionInpaintPipeline
from env import DEVICE
import torch

class SegmentationManager:
    def __init__(self, sam_backbone = "sam_vit_h", g_dino_backbone = "swin_t", device = DEVICE):
        g_dino_model_config_path = "GroundingDINO/groundingdino/config"
        g_dino_checkpoint_path = "weights/GroundingDINO"
        sam_checkpoint_path = "weights/SAM"
        self.device = device

        if sam_backbone == "sam_vit_h":
            sam_checkpoint_path = os.path.join(sam_checkpoint_path, "sam_vit_h_4b8939.pth")
        elif sam_backbone == "sam_vit_b":
            sam_checkpoint_path = os.path.join(sam_checkpoint_path, "sam_vit_b_01ec64.pth")
        elif sam_backbone == "sam_vit_l":
            sam_checkpoint_path = os.path.join(sam_checkpoint_path, "sam_vit_l_0b3195.pth")

        if g_dino_backbone == "swin_t":
            g_dino_model_config_path = os.path.join(g_dino_model_config_path, "GroundingDINO_SwinT_OGC.py")
            g_dino_checkpoint_path = os.path.join(g_dino_checkpoint_path, "groundingdino_swint_ogc.pth")

        # Define DINO model
        self.ground_dino = load_model(g_dino_model_config_path, g_dino_checkpoint_path, device = self.device)

        # Define SAM model
        self.sam = build_sam(checkpoint=sam_checkpoint_path)
        self.sam.to(device=DEVICE)
        # Define SAM predictor
        self.sam_predictor = SamPredictor(self.sam)

        if DEVICE == 'cpu':
            float_type = torch.float32
        else:
            float_type = torch.float16

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=float_type,
        )

        if DEVICE.type != 'cpu':
            self.pipe = self.pipe.to("cuda")

    def detect_objects_image(self, image_path, text_prompt, box_threshold = 0.35, text_threshold = 0.25, output_path = None):
        image_source, image = load_image(image_path)
        boxes, logits, phrases = predict(
        model = self.ground_dino,
        image = image,
        caption = text_prompt,
        box_threshold = box_threshold,
        text_threshold = text_threshold,
        device = self.device
        )
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        
        if output_path is not None:
            cv2.imwrite(output_path, annotated_frame)

        return boxes, logits, phrases, annotated_frame


seg_m = SegmentationManager()
boxes, logits, phrases, _ = seg_m.detect_objects_image("cat.jpg", "cat", output_path = "cat_annotated.jpg")

print(boxes)
print(logits)
print(phrases)

