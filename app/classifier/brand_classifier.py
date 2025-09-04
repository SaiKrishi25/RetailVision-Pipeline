from typing import List, Tuple
import numpy as np
import cv2
from PIL import Image
import torch

class BrandClassifier:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None
        
        # Define common retail brands for classification (simplified labels work better)
        self.brand_labels = [
            "Pantene",
            "Dove", 
            "Head and Shoulders",
            "Colgate",
            "Listerine",
            "Nivea",
            "Loreal",
            "Garnier",
            "Johnsons",
            "Vaseline",
            "generic product"  # Fallback for unrecognized products
        ]
        
        # Brand name mapping (for cleaner group names)
        self.brand_mapping = {
            "Pantene": "Pantene",
            "Dove": "Dove",
            "Head and Shoulders": "Head & Shoulders", 
            "Colgate": "Colgate",
            "Listerine": "Listerine",
            "Nivea": "Nivea",
            "Loreal": "L'Oreal",
            "Garnier": "Garnier",
            "Johnsons": "Johnson's",
            "Vaseline": "Vaseline",
            "generic product": "Unknown"
        }
        
        self._load_model()
    
    def _load_model(self):
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            print("[BrandClassifier] Loading CLIP model...")
            model_name = "openai/clip-vit-base-patch32"
            
            self._processor = CLIPProcessor.from_pretrained(model_name)
            self._model = CLIPModel.from_pretrained(model_name)
            self._model.to(self.device)
            self._model.eval()
            
            print(f"[BrandClassifier] CLIP model loaded on {self.device}")
            
        except Exception as e:
            print(f"[BrandClassifier] Failed to load CLIP model ({e}). Using fallback.")
            self._model = None
            self._processor = None
    
    def classify_brand(self, image_crop: np.ndarray, confidence_threshold: float = 0.15) -> Tuple[str, float]:
        if self._model is None or self._processor is None:
            return "Unknown", 0.0
        
        try:
            # Convert numpy array to PIL Image
            if len(image_crop.shape) == 3 and image_crop.shape[2] == 3:
                # BGR to RGB conversion
                image_crop_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_crop_rgb)
            else:
                return "Unknown", 0.0
            
            # Preprocess inputs
            inputs = self._processor(
                text=self.brand_labels,
                images=pil_image,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)[0]
            
            # Get best prediction
            best_idx = probs.argmax().item()
            confidence = float(probs[best_idx])
            predicted_label = self.brand_labels[best_idx]
            
            # Map to clean brand name
            brand_name = self.brand_mapping.get(predicted_label, "Unknown")
            
            # Apply confidence threshold
            if confidence < confidence_threshold:
                return "Unknown", confidence
            
            return brand_name, confidence
            
        except Exception as e:
            print(f"[BrandClassifier] Classification failed ({e})")
            return "Unknown", 0.0
    
    def classify_batch(self, image_crops: List[np.ndarray]) -> List[Tuple[str, float]]:
        results = []
        for crop in image_crops:
            brand, confidence = self.classify_brand(crop)
            results.append((brand, confidence))
        return results
