from typing import List, Dict
from PIL import Image
import torch

class ProductDetector:
    def __init__(self, model_name: str = "isalia99/detr-resnet-50-sku110k", confidence_threshold: float = 0.5):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self._processor = None
        self._model = None
        
        try:
            from transformers import DetrImageProcessor, DetrForObjectDetection
            print(f"[DETR-SKU110k] Loading model: {model_name}")
            self._processor = DetrImageProcessor.from_pretrained(model_name)
            self._model = DetrForObjectDetection.from_pretrained(model_name)
            self._model.eval()
            
            # Move to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(self.device)
            
            print(f"[DETR-SKU110k] Model loaded successfully on {self.device}")
            
        except Exception as e:
            self._processor = None
            self._model = None
            print(f"[DETR-SKU110k] Failed to load model ({e}). Using dummy detector.")

    def detect(self, image_path: str) -> List[Dict]:
        if self._model is None or self._processor is None:
            return self._dummy_detection(image_path)
        
        try:
            # Load and preprocess image (exact original implementation)
            image = Image.open(image_path).convert("RGB")
            inputs = self._processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self._model(**inputs)
            
            # Post-process results (restored original working version)
            target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
            results = self._processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=self.confidence_threshold
            )[0]
            
            # Convert to our format with edge filtering
            detections = []
            image_width, image_height = image.size
            
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                x1, y1, x2, y2 = box.tolist()
                
                # Filter out edge/half-visible products
                if self._is_fully_visible(x1, y1, x2, y2, image_width, image_height):
                    detections.append({
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "score": float(score.item()),
                        "class": "product"  
                    })
            
            return detections
        except Exception as e:
            print(f"[DETR-SKU110k] Detection failed ({e}). Using dummy fallback.")
            return self._dummy_detection(image_path)
    
    def _is_fully_visible(self, x1: float, y1: float, x2: float, y2: float, 
                         img_width: int, img_height: int) -> bool:
        margin = max(15, min(img_width, img_height) * 0.015)
        width = x2 - x1
        height = y2 - y1
        
        # Calculate how much of the product is visible
        visible_width = min(x2, img_width) - max(x1, 0)
        visible_height = min(y2, img_height) - max(y1, 0)
        visible_area_ratio = (visible_width * visible_height) / (width * height)
        if visible_area_ratio < 0.7:
            return False
        
        min_size = max(20, min(img_width, img_height) * 0.015)
        if visible_width < min_size or visible_height < min_size:
            return False
        if visible_width > 0 and visible_height > 0:
            aspect_ratio = visible_width / visible_height
            if aspect_ratio > 6 or aspect_ratio < 0.15:
                return False
            
        return True

    def _dummy_detection(self, image_path: str) -> List[Dict]:
        import cv2
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        cx, cy = w//2, h//2
        bw, bh = int(w*0.3), int(h*0.3)
        x1, y1 = max(0, cx - bw//2), max(0, cy - bh//2)
        x2, y2 = min(w-1, x1 + bw), min(h-1, y1 + bh)
        return [{
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "score": 0.50,
            "class": "product"
        }]
