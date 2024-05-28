import os
import cv2
import numpy as np
import torch
import torchvision
import supervision as sv

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from groundingdino.util.inference import Model
from utils import helpers
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from scipy.stats import hmean
import pandas as pd
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class SegmentationManager:
#     def __init__(self, sam_type='vit_h', checkpoint_path=None, return_prompts=False, device=DEVICE):
#         self.device = device

#         self.lang_sam = LangSAM(sam_type=sam_type, ckpt_path=checkpoint_path, return_prompts=return_prompts)
#     def lang_segment(self, image_path, text_prompt, box_threshold, text_threshold):
#         image_pil = Image.open(image_path).convert("RGB")
#         masks, boxes, phrases, logits = self.lang_sam.predict(image_pil, text_prompt=text_prompt, text_threshold=text_threshold, box_threshold=box_threshold)
#         masks = masks.cpu().numpy()
#         boxes = boxes.cpu().numpy()
#         phrases = phrases.cpu().numpy()
#         logits = logits.cpu().numpy()
#         return sv.Detections(xyxy=boxes, confidence=logits, class_id=phrases, mask=masks)
    
#     def create_annotation_matrix(self, dataset_path, classes, box_threshold, text_threshold, agg=hmean, run=None, output_namespaces=None):
#         """
#         Create an annotation matrix for a dataset.

#         Parameters:
#         - dataset_path (str): Path to the dataset.
#         - classes (list): List of classes.

#         Returns:
#         - np.ndarray: Annotation matrix with shape (num_images, num_classes).
#         """
#         # Import image base paths
#         image_ids = sorted(os.listdir(dataset_path))

#         # Create a DataFrame full of zeros
#         annotation_matrix = pd.DataFrame(data=np.zeros((len(image_ids), len(classes))), index=image_ids, columns=classes)

#         for image_name in tqdm(image_ids, desc="Creating annotation matrix", unit="image"):
#             image_path = os.path.join(dataset_path, image_name)

#             # Apply the grounding SAM pipeline
#             detections = self.lang_segment(image_path, ". ".join(classes), box_threshold=box_threshold, text_threshold=text_threshold)
#             if run is not None and output_namespaces["detections"] is not None:
#                 run.log_files(data = detections, namespace=os.path.join(output_namespaces["detections"], image_name), extension="pkl", wait=True)
#             # Calculate the harmonic mean of confidences
#             confidences = detections.confidence
#             class_ids = [classes[id] for id in detections.class_id]
#             harmonic_means = self.aggregator(class_ids, confidences, agg=agg)
#             annotation_matrix.loc[image_name, harmonic_means[:, 0]] = harmonic_means[:, 1]
#         if run is not None and output_namespaces["annotation_matrix"] is not None:
#             run.log_dataframe(dataframe=annotation_matrix, namespace=output_namespaces["annotation_matrix"], csv_format=True, index=True, wait=True)

#         return annotation_matrix

#     def occ_proba_disjoint_tensor(self, matrix: pd.DataFrame = None, apply_on_annotation_matrix = True, run=None, output_namespaces=None, **kwargs):
#         """
#         Calculate the occurrence probability disjoint matrix.

#         Parameters:
#         - matrix (np.ndarray): Annotation matrix with shape (num_images, num_classes).

#         Returns:
#         - np.ndarray: Occurrence probability disjoint matrix with shape (num_classes, num_classes).
#         """
#         dataset_path = kwargs.get("dataset_path", None)
#         classes = kwargs.get("classes", None)
#         box_threshold = kwargs.get("box_threshold", 0.25)
#         text_threshold = kwargs.get("text_threshold", 0.25)

#         agg = kwargs.get("agg", hmean)
#         annotation_matrix_processing = kwargs.get("annotation_matrix_processing", None)
#         annotation_matrix_processing_args = kwargs.get("annotation_matrix_processing_args", {})
#         if apply_on_annotation_matrix:
#             assert matrix is None, "Annotation matrix should not be provided."
#             matrix = self.create_annotation_matrix(dataset_path=dataset_path, 
#                                                    classes=classes, 
#                                                    box_threshold=box_threshold, 
#                                                    text_threshold=text_threshold,
#                                                    agg=agg,
#                                                    run=run,
#                                                    output_namespaces=output_namespaces)
#         if annotation_matrix_processing is not None:
#             matrix = annotation_matrix_processing(matrix, **annotation_matrix_processing_args)

#         classes = matrix.columns
#         num_rows, num_cols = matrix.shape

#         tensor = np.zeros(shape=(num_cols, num_rows, num_rows))
#         for i, c in enumerate(classes):
#             column = matrix.loc[:, c].values
#             column = np.array([float(val) for val in column]).reshape(len(column), 1)

#             # The result is square rooted to normalize the values
#             mat = np.sqrt(column * column.T)       
#             tensor[i] = mat
#         if run is not None and output_namespaces["adjacency_tensor"] is not None:
#             run.log_files(data=tensor, namespace=output_namespaces["adjacency_tensor"], extension="pkl", wait=True)

#         return {"adjacency_tensor" : tensor, "index": matrix.index, "columns": classes, "annotation_matrix": matrix}


class SegmentationManager:
    def __init__(self, sam_backbone="sam_vit_h", g_dino_backbone="swin_t", device=DEVICE):
        self.device = device
        
        # Paths for checkpoints and configs
        g_dino_model_config_path = "GroundingDINO/groundingdino/config"
        g_dino_checkpoint_path = "weights/GroundingDINO"
        sam_checkpoint_path = "weights/SAM"
        sam_type = None


        helpers.create_hierarchy(g_dino_checkpoint_path)
        helpers.create_hierarchy(sam_checkpoint_path)

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
            config_name = "GroundingDINO_SwinT_OGC.py"
            g_dino_checkpoint_path = os.path.join(g_dino_checkpoint_path, "groundingdino_swint_ogc.pth")
            g_dino_model_config_path = os.path.join(g_dino_model_config_path, config_name)


            if not os.path.exists(g_dino_checkpoint_path):
                os.system(f"wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth")
                os.system(f"mv groundingdino_swint_ogc.pth {g_dino_checkpoint_path}")

        # Initialize models
        self.grounding_dino_model = Model(model_config_path=g_dino_model_config_path, 
                                          model_checkpoint_path=g_dino_checkpoint_path,
                                          device = self.device)

        self.sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint_path)
        self.sam.to(device=self.device)
        self.sam_predictor = SamPredictor(self.sam)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)

    def load_image(self, image_path):
        return cv2.imread(image_path)

    def set_mask_generator_(self, **kwargs):
        self.mask_generator = SamAutomaticMaskGenerator(self.sam, **kwargs)

    def detect_objects(self, image, classes, box_threshold=0.25, text_threshold=0.25):
        detections = self.grounding_dino_model.predict_with_classes(
            image=image,
            classes=classes,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        return detections
    
    def iou(self, box1, box2):
        """
        Compute the Intersection over Union (IoU) of two bounding boxes.

        Parameters:
        - box1, box2 (list or np.array): Bounding boxes in the format [x1, y1, x2, y2].

        Returns:
        - float: IoU of the two bounding boxes.
        """
        # Coordinates of the intersection rectangle
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        # Compute the area of the intersection rectangle
        inter_width = max(0, x2_inter - x1_inter + 1)
        inter_height = max(0, y2_inter - y1_inter + 1)
        inter_area = inter_width * inter_height

        # Compute the area of both bounding boxes
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        # Compute the IoU by taking the intersection area and dividing it by the union area
        union_area = box1_area + box2_area - inter_area
        iou_value = inter_area / union_area

        return iou_value

    def nms(self, detections, nms_threshold=0.5):
        bboxes = detections.xyxy
        confidences = detections.confidence
        class_ids = detections.class_id

        to_drop = []
        for i, bbox_i in enumerate(bboxes):
            for j, bbox_j in enumerate(bboxes):
                if i != j and class_ids[i]==class_ids[j] and self.iou(bbox_i, bbox_j) > nms_threshold:
                    if confidences[i] > confidences[j]:
                        to_drop.append(j)
                    else:
                        to_drop.append(i)

        to_keep = [i for i in range(len(bboxes)) if i not in to_drop]
        detections.xyxy = detections.xyxy[to_keep]
        detections.confidence = detections.confidence[to_keep]
        detections.class_id = detections.class_id[to_keep]

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
        # idx = first_occurrences_indices(detections.class_id)

        # detections.xyxy = detections.xyxy[idx]
        # detections.confidence = detections.confidence[idx]
        # detections.class_id = detections.class_id[idx]

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
        # Find indices of None values
        none_indices = np.where(class_ids == None)[0]

        # Remove None values
        class_ids = np.delete(class_ids, none_indices)
        confidences = np.delete(confidences, none_indices)
        detections.xyxy = np.delete(detections.xyxy, none_indices, axis=0)
        detections.mask = np.delete(detections.mask, none_indices, axis=0)
        detections.class_id = class_ids
        detections.confidence = confidences

        labels = [f"{classes[class_id]} {confidence:0.2f}" for class_id, confidence in zip(class_ids, confidences)]

        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        return annotated_image

    def grounding_sam(self, source_image_path, classes, box_threshold=0.25, text_threshold=0.25, nms_threshold=0.8, output_path=None):
        # Load image
        image = self.load_image(source_image_path)

        # Detect objects
        detections = self.detect_objects(image, classes, box_threshold, text_threshold)
        # Apply NMS
        detections = self.nms(detections, nms_threshold)

        # Convert detections to masks
        detections.mask = self.segment(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections.xyxy)
  

        # Annotate image with segmented masks
        annotated_image = self.annotate_image(image, detections, classes)

        if output_path is not None:
            self.save_image(annotated_image, output_path)

        return detections, annotated_image
    
    def aggregator(self, classes, confidences, agg=hmean):
        """
        Calculate the harmonic mean of confidences for each class.

        Parameters:
        - classes (np.ndarray): 1D array containing class IDs.
        - confidences (np.ndarray): 1D array containing confidences.

        Returns:
        - np.ndarray: 2D array with class_id and their corresponding harmonic means of confidences.
        """
        classes = np.array(classes)
        confidences = np.array(confidences)
        unique_classes = np.unique(classes)
        result = []

        for cls in unique_classes:
            class_confidences = confidences[classes == cls]
            harmonic_mean_value = agg(class_confidences)
            result.append([cls, harmonic_mean_value])

        return np.array(result)

    def create_annotation_matrix(self, dataset_path, classes, box_threshold, text_threshold, nms_threshold, agg=hmean, run=None, output_namespaces=None):
        """
        Create an annotation matrix for a dataset.

        Parameters:
        - dataset_path (str): Path to the dataset.
        - classes (list): List of classes.

        Returns:
        - np.ndarray: Annotation matrix with shape (num_images, num_classes).
        """
        # Import image base paths
        image_ids = sorted(os.listdir(dataset_path))

        # Create a DataFrame full of zeros
        annotation_matrix = pd.DataFrame(data=np.zeros((len(image_ids), len(classes))), index=image_ids, columns=classes)

        for image_name in tqdm(image_ids, desc="Creating annotation matrix", unit="image"):
            image_path = os.path.join(dataset_path, image_name)

            # Apply the grounding SAM pipeline
            detections, annotated_image = self.grounding_sam(image_path, 
                                                             classes, 
                                                             box_threshold=box_threshold, 
                                                             text_threshold=text_threshold, 
                                                             nms_threshold=nms_threshold)
        
            if run is not None and output_namespaces["annotated_images"] is not None:
                # Define a transform to convert the image to tensor
                transform = transforms.ToTensor()

                # Convert the image to PyTorch tensor
                rgb_image = transform(annotated_image)
                rgb_image = rgb_image[[2, 1, 0], :, :]
                run.log_tensors(tensors=[rgb_image], paths=[os.path.join(output_namespaces["annotated_images"], image_name)], on_series=False)
            if run is not None and output_namespaces["detections"] is not None:
                run.log_files(data = detections, namespace=os.path.join(output_namespaces["detections"], image_name), extension="pkl", wait=True)
            # Calculate the harmonic mean of confidences
            confidences = detections.confidence
            class_ids = [classes[id] for id in detections.class_id]
            harmonic_means = self.aggregator(class_ids, confidences, agg=agg)
            annotation_matrix.loc[image_name, harmonic_means[:, 0]] = harmonic_means[:, 1]
        if run is not None and output_namespaces["annotation_matrix"] is not None:
            run.log_dataframe(dataframe=annotation_matrix, namespace=output_namespaces["annotation_matrix"], csv_format=True, index=True, wait=True)

        return annotation_matrix
    
    def occ_proba_disjoint_tensor(self, matrix: pd.DataFrame = None, apply_on_annotation_matrix = True, run=None, output_namespaces=None, **kwargs):
        """
        Calculate the occurrence probability disjoint matrix.

        Parameters:
        - matrix (np.ndarray): Annotation matrix with shape (num_images, num_classes).

        Returns:
        - np.ndarray: Occurrence probability disjoint matrix with shape (num_classes, num_classes).
        """
        dataset_path = kwargs.get("dataset_path", None)
        classes = kwargs.get("classes", None)
        box_threshold = kwargs.get("box_threshold", 0.25)
        text_threshold = kwargs.get("text_threshold", 0.25)
        nms_threshold = kwargs.get("nms_threshold", 0.8)
        agg = kwargs.get("agg", hmean)
        annotation_matrix_processing = kwargs.get("annotation_matrix_processing", None)
        annotation_matrix_processing_args = kwargs.get("annotation_matrix_processing_args", {})
        if apply_on_annotation_matrix:
            assert matrix is None, "Annotation matrix should not be provided."
            matrix = self.create_annotation_matrix(dataset_path=dataset_path, 
                                                   classes=classes, 
                                                   box_threshold=box_threshold, 
                                                   text_threshold=text_threshold,
                                                   nms_threshold=nms_threshold, 
                                                   agg=agg,
                                                   run=run,
                                                   output_namespaces=output_namespaces)
        if annotation_matrix_processing is not None:
            matrix = annotation_matrix_processing(matrix, **annotation_matrix_processing_args)

        classes = matrix.columns
        num_rows, num_cols = matrix.shape

        tensor = np.zeros(shape=(num_cols, num_rows, num_rows))
        for i, c in enumerate(classes):
            column = matrix.loc[:, c].values
            column = np.array([float(val) for val in column]).reshape(len(column), 1)

            # The result is square rooted to normalize the values
            mat = np.sqrt(column * column.T)       
            tensor[i] = mat
        if run is not None and output_namespaces["adjacency_tensor"] is not None:
            run.log_files(data=tensor, namespace=output_namespaces["adjacency_tensor"], extension="pkl", wait=True)

        return {"adjacency_tensor" : tensor, "index": matrix.index, "columns": classes, "annotation_matrix": matrix}

    def save_image(self, image, path):
        cv2.imwrite(path, image)

    def auto_segmentation(self, img_path, img_size=(1024, 1024), run=None, output_namespaces=None, **kwargs):
        def show_anns(anns):
            if len(anns) == 0:
                return
            sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

            img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
            img[:,:,3] = 0
            for ann in sorted_anns:
                m = ann['segmentation']
                color_mask = np.concatenate([np.random.random(3), [0.35]])
                img[m] = color_mask
            return img
            
        # Read the image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, img_size)

        if kwargs != {}:
            self.set_mask_generator_(**kwargs)

        masks = self.mask_generator.generate(image)
        annot_img = show_anns(masks)

        if run is not None:
            if output_namespaces["annotated_images"] is not None:
                # Define a transform to convert the image to tensor
                transform = transforms.ToTensor()

                # Convert the image to PyTorch tensor
                rgb_image = transform(image)
                rgb_image = rgb_image[[2, 1, 0], :, :]
                run.log_tensors(tensors=[rgb_image], paths=[os.path.join(output_namespaces["annotated_images"], img_path.split("/")[-1])], on_series=False)
            
            if output_namespaces["detections"] is not None:
                image_name = img_path.split("/")[-1]
                image_name = image_name.split(".")[0]
                run.log_files(data = masks, namespace=os.path.join(output_namespaces["detections"], image_name), extension="pkl", wait=True)

        return masks, annot_img
        
# # # Example usage
# if __name__ == "__main__":
#     SOURCE_IMAGE_PATH = "G0041951.JPG"
#     CLASSES = ["person", "banner", "finger", "hand", "foot", "glasses", "desert", "sky", "clouds"]
#     BOX_THRESHOLD = 0.3
#     TEXT_THRESHOLD = 0.25
#     NMS_THRESHOLD = 0.3

#     manager = SegmentationManager()

#     # Apply the grounding SAM pipeline
#     # detections = manager.grounding_sam(SOURCE_IMAGE_PATH, CLASSES, BOX_THRESHOLD, TEXT_THRESHOLD, NMS_THRESHOLD, "grounded_sam_annotated_image.jpg")

#     # Create an annotation matrix for a dataset
#     dataset_path = "test_dataset"
#     annotation_matrix = manager.occ_proba_disjoint_tensor(matrix = None, 
#                                                           apply_on_annotation_matrix=True, 
#                                                           dataset_path=dataset_path, 
#                                                           classes=CLASSES, 
#                                                           box_threshold=BOX_THRESHOLD, 
#                                                           text_threshold=TEXT_THRESHOLD, 
#                                                           nms_threshold=NMS_THRESHOLD)

