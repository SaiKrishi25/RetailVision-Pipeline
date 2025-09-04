
from typing import List, Dict
import numpy as np
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as T
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from app.classifier.brand_classifier import BrandClassifier

class ProductGrouper:
    def __init__(self, embedding_model: str = "resnet50", device: str = None, sim_threshold: float = 0.45):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sim_threshold = sim_threshold
        self.embedding_model = embedding_model
        # Initialize CLIP-based brand classifier
        self.brand_classifier = BrandClassifier(device=self.device)
        # Initialize visual feature extractor
        self._load_feature_extractor()
        # Initialize clustering
        self.clusterer = DBSCAN(eps=0.4, min_samples=2, metric='cosine')
        # Brand to group ID mapping (will be dynamic)
        self.brand_to_group = {}
        self.group_counter = 1

    def _load_feature_extractor(self):
        try:
            print(f"[ProductGrouper] Loading {self.embedding_model} for visual features...")
            
            # Load pretrained ResNet50
            self.feature_model = models.resnet50(pretrained=True)
            self.feature_model = torch.nn.Sequential(*list(self.feature_model.children())[:-1])
            self.feature_model.to(self.device)
            self.feature_model.eval()
            
            # Define image transforms
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            print(f"[ProductGrouper] Feature extractor loaded on {self.device}")
        except Exception as e:
            print(f"[ProductGrouper] Failed to load feature extractor: {e}")
            self.feature_model = None
            self.transform = None

    def extract_visual_features(self, image_crops: List[np.ndarray]) -> np.ndarray:
        if self.feature_model is None:
            return np.random.rand(len(image_crops), 512)
        features = []
        for crop in image_crops:
            try:
                if len(crop.shape) == 3 and crop.shape[2] == 3:
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                else:
                    crop_rgb = crop
                # Transform and extract features
                input_tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feature = self.feature_model(input_tensor)
                    feature = feature.squeeze().cpu().numpy()
                    feature = feature / (np.linalg.norm(feature) + 1e-8)
                    features.append(feature)
            except Exception as e:
                print(f"[ProductGrouper] Feature extraction failed for crop: {e}")
                features.append(np.random.rand(2048))
        return np.array(features)

    def perform_visual_clustering(self, features: np.ndarray, brand_labels: List[str]) -> List[int]:
        if len(features) < 2:
            return [0] * len(features)
        # Group by brands first
        brand_groups = {}
        for i, brand in enumerate(brand_labels):
            if brand not in brand_groups:
                brand_groups[brand] = []
            brand_groups[brand].append(i)
        cluster_labels = [-1] * len(features)
        cluster_id = 0
        
        # Cluster within each brand group
        for brand, indices in brand_groups.items():
            if len(indices) == 1:
                cluster_labels[indices[0]] = cluster_id
                cluster_id += 1
            else:
                brand_features = features[indices]
                try:
                    # Use DBSCAN for clustering
                    sub_clusters = self.clusterer.fit_predict(brand_features)
                    unique_clusters = set(sub_clusters)
                    if -1 in unique_clusters:
                        unique_clusters.remove(-1)
                    # Assign cluster IDs
                    cluster_mapping = {}
                    for unique_cluster in unique_clusters:
                        cluster_mapping[unique_cluster] = cluster_id
                        cluster_id += 1
                    # Assign labels
                    for j, sub_cluster in enumerate(sub_clusters):
                        original_idx = indices[j]
                        if sub_cluster == -1:
                            cluster_labels[original_idx] = cluster_id
                            cluster_id += 1
                        else:
                            cluster_labels[original_idx] = cluster_mapping[sub_cluster]
                except Exception as e:
                    print(f"[ProductGrouper] Clustering failed for brand {brand}: {e}")
                    for j, idx in enumerate(indices):
                        cluster_labels[idx] = cluster_id + j
                    cluster_id += len(indices)
        return cluster_labels

    def refine_grouping_with_similarity(self, detections: List[Dict], features: np.ndarray) -> List[Dict]:
        if len(detections) < 2:
            return detections
        # Group detections by current group_id
        groups = {}
        for i, det in enumerate(detections):
            group_id = det["group_id"]
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append((i, det))
        refined_detections = []
        new_group_counter = 1
        
        for group_id, group_items in groups.items():
            if len(group_items) == 1:
                # Single item keeps its group
                det = group_items[0][1].copy()
                det["group_id"] = f"G{new_group_counter}"
                det["visual_similarity_score"] = 1.0
                refined_detections.append(det)
                new_group_counter += 1
            else:
                # Multiple items - check visual similarity
                indices = [item[0] for item in group_items]
                group_features = features[indices]
                # Calculate pairwise similarities
                similarities = cosine_similarity(group_features)
                # Use similarity threshold to further subdivide
                refined_clusters = []
                unassigned = list(range(len(group_items)))
                while unassigned:
                    # Start new cluster with first unassigned item
                    current_cluster = [unassigned.pop(0)]
                    # Find similar items
                    i = 0
                    while i < len(unassigned):
                        candidate_idx = unassigned[i]
                        # Check similarity with all items in current cluster
                        max_similarity = 0.0
                        for cluster_idx in current_cluster:
                            sim = similarities[cluster_idx][candidate_idx]
                            max_similarity = max(max_similarity, sim)
                        
                        if max_similarity > self.sim_threshold:
                            current_cluster.append(unassigned.pop(i))
                        else:
                            i += 1
                    refined_clusters.append(current_cluster)
                # Assign new group IDs
                for cluster in refined_clusters:
                    group_name = f"G{new_group_counter}"
                    avg_similarity = 0.0 
                    if len(cluster) > 1:
                        # Calculate average similarity within cluster
                        sim_scores = []
                        for i in range(len(cluster)):
                            for j in range(i+1, len(cluster)):
                                sim_scores.append(similarities[cluster[i]][cluster[j]])
                        avg_similarity = np.mean(sim_scores) if sim_scores else 1.0
                    else:
                        avg_similarity = 1.0
                    for cluster_idx in cluster:
                        original_det = group_items[cluster_idx][1]  # Extract the detection dict from the tuple
                        refined_det = original_det.copy()
                        refined_det["group_id"] = group_name
                        refined_det["visual_similarity_score"] = float(avg_similarity)
                        refined_detections.append(refined_det)
                    new_group_counter += 1
        return refined_detections

    def group(self, image_path: str, detections: List[Dict]) -> List[Dict]:
        if not detections:
            return []
        img = cv2.imread(image_path)
        if img is None:
            print(f"[ProductGrouper] Error: Could not load image from {image_path}")
            return []
        print(f"[ProductGrouper] Processing {len(detections)} detections with enhanced grouping...")
        
        # Step 1: Extract crops and classify brands
        crops = []
        brand_results = []
        valid_detections = []
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            crop = img[max(y1,0):max(y2,0), max(x1,0):max(x2,0)]
            if crop.size == 0:
                brand_name = "Unknown"
                confidence = 0.0
            else:
                brand_name, confidence = self.brand_classifier.classify_brand(crop)
                crops.append(crop)
            brand_results.append((brand_name, confidence))
            valid_detections.append(det)
        
        # Step 2: Extract visual features for valid crops
        if crops:
            visual_features = self.extract_visual_features(crops)
        else:
            visual_features = np.array([])
        
        # Step 3: Initial brand-based grouping
        initial_grouped = []
        self.brand_to_group = {}
        self.group_counter = 1
        for i, det in enumerate(valid_detections):
            brand_name, confidence = brand_results[i]
            if brand_name not in self.brand_to_group:
                self.brand_to_group[brand_name] = f"G{self.group_counter}"
                self.group_counter += 1
            grouped_det = det.copy()
            grouped_det["group_id"] = self.brand_to_group[brand_name]
            grouped_det["brand_name"] = brand_name
            grouped_det["brand_confidence"] = confidence
            initial_grouped.append(grouped_det) 
        # Step 4: Refine grouping using visual similarity
        if len(crops) > 0 and visual_features.size > 0:
            print(f"[ProductGrouper] Refining groups using visual similarity...")
            final_grouped = self.refine_grouping_with_similarity(initial_grouped, visual_features)
        else:
            final_grouped = initial_grouped
        # Step 5: Add grouping statistics
        group_stats = {}
        for det in final_grouped:
            group_id = det["group_id"]
            brand_name = det["brand_name"] 
            if group_id not in group_stats:
                group_stats[group_id] = {"count": 0, "brand": brand_name, "avg_confidence": 0.0}
            group_stats[group_id]["count"] += 1
            group_stats[group_id]["avg_confidence"] += det["brand_confidence"]
        
        # Calculate average confidences
        for group_id in group_stats:
            group_stats[group_id]["avg_confidence"] /= group_stats[group_id]["count"]
        for det in final_grouped:
            group_id = det["group_id"]
            det["group_size"] = group_stats[group_id]["count"]
            det["group_avg_confidence"] = group_stats[group_id]["avg_confidence"]
        print(f"[ProductGrouper] Created {len(group_stats)} groups from {len(detections)} detections")
        return final_grouped
