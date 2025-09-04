
# RetailVision Pipeline – AI-Powered Retail Product Analysis

This repository implements a production-ready AI pipeline for comprehensive retail product analysis:
- **Flask** webserver (upload UI + `/api/infer`)
- **DETR-ResNet-50** detector trained specifically on SKU-110k dataset (58.9% mAP)
- **CLIP-based brand classification** for 11 major retail brands (Dove, Pantene, Colgate, etc.)
- **Intelligent product grouping** (ResNet50 visual features + brand-aware clustering)
- **Color-coded visualization** with brand labels saved to `app/static/outputs/<id>.jpg`

> **Enterprise-ready**: Combines state-of-the-art DETR detection, CLIP brand recognition, and smart visual grouping for complete retail analytics.

---

## Project Structure
```
app/
  detector/model.py         # DETR-ResNet-50 (SKU-110k trained)
  classifier/
    brand_classifier.py     # CLIP-based brand recognition (11 brands)
  grouper/group.py          # Brand-aware clustering + ResNet50 visual features
  utils/
    visualize.py            # Color-coded boxes with brand labels
    schema.py               # Enhanced JSON response schema
  main.py                   # Flask server with brand classification
  templates/index.html      # Upload UI with brand display
  static/
    uploads/                # Uploaded images
    outputs/                # Annotated visualizations with brands
requirements.txt            # Includes transformers, CLIP dependencies
Dockerfile                  # Production container setup
run.bat                     # Windows launcher
sample_images/              # Test retail shelf images
DEPLOYMENT_DOCUMENTATION.md # Comprehensive deployment guide
```

---

## Quickstart (Local)

### Git Clone
```bash
git clone https://github.com/SaiKrishi25/RetailVision-Pipeline.git
cd RetailVision-Pipeline
```

```bash

# 1. Setup environment
python -m venv .venv && source .venv/bin/activate  # or Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. Run the pipeline
.\run.bat  # Windows
# or python -m flask run --host=0.0.0.0 --port=8000

# 3. Open browser to http://localhost:8000
```

**First run**: Downloads DETR model (~167MB) and CLIP model (~600MB) from Hugging Face
**Subsequent runs**: Instant startup with cached models

Upload a retail shelf image and inspect the enhanced JSON response with brand classification + color-coded visualization!

---

## JSON I/O Formats

### Request
- `multipart/form-data` with a file field named **`image`**

### Response
```json
{
  "image_id": "a1b2c3d4",
  "detections": [
    {
      "bbox": [x1, y1, x2, y2],
      "score": 0.92,
      "class": "product",
      "group_id": "G1",
      "brand_name": "Dove",
      "brand_confidence": 0.87
    }
  ],
  "original_image": "/uploads/a1b2c3d4.jpg",
  "visualization": "/outputs/a1b2c3d4.jpg"
}
```

---

## Docker (production-style)
```bash
docker build -t infilect-ai-pipeline .
docker run --rm -p 8000:8000 infilect-ai-pipeline
```

---

## Design Notes (Microservice-friendly)
- **Detector** and **Grouper** are isolated modules → can be broken out behind REST/gRPC later.
- For scalability, run multiple Gunicorn workers behind **Nginx** or a managed LB.
- Use **Redis/RabbitMQ** + **Celery** if you later need async batch processing.
- Persist outputs to S3/GCS/Azure Blob; keep URLs in responses.

---

## Model Architecture

### Detection Model: DETR-ResNet-50-SKU110k
- **Architecture**: Detection Transformer (DETR) with ResNet-50 backbone
- **Training Data**: SKU-110k dataset (11,762 images, 1.7M product annotations)
- **Performance**: 58.9% mAP on SKU-110k validation set
- **Specialization**: Optimized for dense retail product detection
- **Input**: 800px images (automatically resized)
- **Output**: Up to 400 product detections per image

### Brand Classification Model: CLIP-ViT-B/32
- **Architecture**: Vision Transformer (ViT) with text encoder
- **Model**: OpenAI CLIP (openai/clip-vit-base-patch32)
- **Brands Supported**: 11 major retail brands (Pantene, Dove, Head & Shoulders, Colgate, Listerine, Nivea, L'Oreal, Garnier, Johnson's, Vaseline, Unknown)
- **Input**: Cropped product regions from detection
- **Confidence Threshold**: 0.15 (configurable)
- **Purpose**: Identifies specific brand names for accurate grouping

### Grouping Model: Brand-Aware Visual Clustering
- **Primary**: CLIP brand classification for initial grouping
- **Secondary**: ResNet50 (ImageNet pretrained) visual features
- **Clustering**: DBSCAN within brand groups + cosine similarity refinement
- **Threshold**: 0.45 (configurable via `GROUPING_THRESHOLD`)
- **Purpose**: Groups products by brand first, then visual similarity within brands

### Configuration Options
```bash
# Adjust detection confidence (default: 0.5)
set CONFIDENCE_THRESHOLD=0.3

# Adjust brand classification confidence (default: 0.15)
set BRAND_CONFIDENCE_THRESHOLD=0.2  # stricter brand classification

# Adjust visual grouping sensitivity (default: 0.45)
set GROUPING_THRESHOLD=0.6  # stricter visual grouping within brands
set GROUPING_THRESHOLD=0.3  # looser visual grouping within brands

# Override models (optional)
set DETECTOR_MODEL=isalia99/detr-resnet-50-sku110k
set EMBEDDING_MODEL=resnet50
```

---

## Assignment Deliverables
- Flask webserver 
- Detection model + grouping 
- Brand classification with CLIP
- Microservice-like modular design 
- JSON input/output defined 
- Color-coded visualizations saved to file 
- Setup & run instructions 

---

## Enhanced Features Implemented
- **CLIP-based brand classifier** for explicit brand identification
- **Brand-aware intelligent grouping** (DBSCAN + visual similarity)
- **11 major retail brands** supported with confidence scoring
- **Enhanced JSON response** with brand information
- **Professional deployment documentation** with performance metrics
