
import os
import uuid
from flask import Flask, request, jsonify, render_template, send_from_directory
from app.detector.model import ProductDetector
from app.grouper.group import ProductGrouper
from app.utils.visualize import draw_grouped_detections
from app.utils.schema import to_response_schema

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "app/static/uploads")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "app/static/outputs")

print(f"[DEBUG] UPLOAD_DIR: {UPLOAD_DIR}")
print(f"[DEBUG] OUTPUT_DIR: {OUTPUT_DIR}")
print(f"[DEBUG] UPLOAD_DIR exists: {os.path.exists(UPLOAD_DIR)}")
print(f"[DEBUG] OUTPUT_DIR exists: {os.path.exists(OUTPUT_DIR)}")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")

# Initialize services (could be separate microservices; here modularized for simplicity)
detector = ProductDetector(
    model_name=os.environ.get("DETECTOR_MODEL", "isalia99/detr-resnet-50-sku110k"),
    confidence_threshold=float(os.environ.get("CONFIDENCE_THRESHOLD", "0.5"))
)
grouper = ProductGrouper(
    embedding_model=os.environ.get("EMBEDDING_MODEL", "resnet50"),
    sim_threshold=float(os.environ.get("GROUPING_THRESHOLD", "0.45"))  
)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/api/infer", methods=["POST"])
def infer():
    # Accept file upload under 'image'
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided under form field 'image'."}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename."}), 400

    image_id = str(uuid.uuid4())[:8]
    in_path = os.path.join(UPLOAD_DIR, f"{image_id}.jpg")
    out_path = os.path.join(OUTPUT_DIR, f"{image_id}.jpg")
    file.save(in_path)

    # Detection
    detections = detector.detect(in_path)
    # Grouping
    grouped = grouper.group(in_path, detections)
    # Visualization
    draw_grouped_detections(in_path, grouped, out_path)
    # JSON response
    response = to_response_schema(image_id=image_id, grouped_detections=grouped, visualization_path=out_path)
    return jsonify(response)

@app.route("/outputs/<filename>")
def outputs(filename):
    abs_output_dir = os.path.abspath(OUTPUT_DIR)
    print(f"[DEBUG] Serving output file: {filename} from {abs_output_dir}")
    file_path = os.path.join(abs_output_dir, filename)
    print(f"[DEBUG] Full file path: {file_path}")
    print(f"[DEBUG] File exists: {os.path.exists(file_path)}")
    return send_from_directory(abs_output_dir, filename, as_attachment=False)

@app.route("/uploads/<filename>")
def uploads(filename):
    abs_upload_dir = os.path.abspath(UPLOAD_DIR)
    print(f"[DEBUG] Serving upload file: {filename} from {abs_upload_dir}")
    file_path = os.path.join(abs_upload_dir, filename)
    print(f"[DEBUG] Full file path: {file_path}")
    print(f"[DEBUG] File exists: {os.path.exists(file_path)}")
    return send_from_directory(abs_upload_dir, filename, as_attachment=False)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=True)
