@echo off
echo ========================================
echo    RetailVision Pipeline - DETR-SKU110k
echo ========================================

set FLASK_APP=app/main.py
set UPLOAD_DIR=app/static/uploads
set OUTPUT_DIR=app/static/outputs
set PORT=8000

REM DETR-SKU110k Configuration
set DETECTOR_MODEL=isalia99/detr-resnet-50-sku110k
set CONFIDENCE_THRESHOLD=0.5
set GROUPING_THRESHOLD=0.50
set EMBEDDING_MODEL=resnet50

echo Configuration:
echo - Detection Model: DETR-ResNet-50 (SKU-110k trained)
echo - Confidence Threshold: %CONFIDENCE_THRESHOLD%
echo - Grouping Threshold: %GROUPING_THRESHOLD%
echo - Port: %PORT%
echo.
echo Note: First run will download the model (~167MB)
echo.

python -m flask run --host=0.0.0.0 --port=%PORT%
