import os
def to_response_schema(image_id, grouped_detections, visualization_path):
    # Normalize path for web route if inside static/outputs
    vis_rel = visualization_path
    if "app/static/outputs/" in visualization_path:
        vis_rel = os.path.basename(visualization_path)
    elif "static/outputs/" in visualization_path:
        vis_rel = os.path.basename(visualization_path)
    else:
        vis_rel = os.path.basename(visualization_path)
    return {
        "image_id": image_id,
        "detections": [
            {
                "bbox": det["bbox"],
                "score": float(det.get("score", 0.0)),
                "class": det.get("class", "product"),
                "group_id": det.get("group_id", "G0"),
                "brand_name": det.get("brand_name", "Unknown"),
                "brand_confidence": float(det.get("brand_confidence", 0.0))
            }
            for det in grouped_detections
        ],
        "original_image": f"/uploads/{image_id}.jpg",
        "visualization": f"/outputs/{vis_rel}"
    }
