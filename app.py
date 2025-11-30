import os
import json
import torch
import torch.nn as nn
import timm
import torchvision.transforms as T
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
# Imports for batch processing
import io
import base64
import zipfile
import tempfile
from werkzeug.utils import secure_filename

# --- Model Definition ---
class InsectModel(nn.Module):
    def __init__(self, num_classes):
        super(InsectModel, self).__init__()
        self.model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=False,
            num_classes=num_classes
        )

    def forward(self, image):
        return self.model(image)

# --- Load Class Mapping from labeled_classes.txt ---
labeled_classes_path = os.path.join(os.path.dirname(__file__), "labeled_classes.txt")
if not os.path.exists(labeled_classes_path):
    raise FileNotFoundError(f"labeled_classes.txt not found: {labeled_classes_path}")

# First, load all pest names by their label number
label_to_name = {}
with open(labeled_classes_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            label = int(parts[0])
            name = parts[1]
            label_to_name[label] = name

# Create the class_names list in the order that ImageFolder would use (alphabetical by folder name)
folder_names_sorted = sorted([str(i) for i in range(1, 41)], key=str)
class_names = [label_to_name.get(int(folder_name), f"Unknown_{folder_name}") for folder_name in folder_names_sorted]

# --- Load Pest Details from JSON (convert list → dict) ---
pest_details_path = os.path.join(os.path.dirname(__file__), "pest.pest_details.json")
if not os.path.exists(pest_details_path):
    raise FileNotFoundError(f"Pest details file not found: {pest_details_path}")

with open(pest_details_path, "r", encoding="utf-8") as f:
    pest_list = json.load(f)

# Convert to dictionary keyed by pest name
pest_info_map = {}
for pest in pest_list:
    pest_name = pest.get("name", "")
    if pest_name:
        pest_info_map[pest_name] = {
            "description": pest.get("description", "No description available."),
            "prevention": pest.get("prevention", "No prevention info available."),
            "pesticides": pest.get("pesticides", "No pesticide info available."),
            "p_image": pest.get("p_image", ""),  # Main pest image
            "pest_image": pest.get("pest_image", [])  # Pesticide images list
        }

# --- Confidence Threshold ---
CONF_THRESHOLD = 0.5

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Model ---
model = InsectModel(num_classes=len(class_names))
model_path = os.path.join(os.path.dirname(__file__), "vit_best.pth")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# --- Image Preprocessing ---
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

# --- Helper functions for batch processing ---
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "gif", "webp"}

def allowed_image(filename):
    """Check if file has an allowed image extension"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_data_url(image_path, max_size=150):
    """Convert image to base64 data URL for display in HTML"""
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            img_data = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_data}"
    except Exception as e:
        print(f"Error converting image to data URL: {e}")
        return None

def get_pesticide_image_data_url(image_filename, max_size=200):
    """
    Correctly finds an image in the 'static/pesticides/' folder and converts it to a base64 data URL.
    """
    try:
        # Build the full path to the pesticide image in the static/pesticides folder
        base_dir = os.path.dirname(__file__)
        # Use os.path.basename to ensure we only have the filename, preventing path issues
        safe_filename = os.path.basename(image_filename)
        full_path = os.path.join(base_dir, 'static', 'pesticides', safe_filename)
        
        # This print helps debug if the path is correct
        print(f"[DEBUG] Looking for pesticide image at: {full_path}")
        
        if os.path.exists(full_path):
            img = Image.open(full_path).convert("RGB")
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            img_data = base64.b64encode(buffer.getvalue()).decode()
            print(f"[DEBUG] Successfully loaded: {full_path}")
            return f"data:image/jpeg;base64,{img_data}"
        else:
            print(f"[WARNING] Pesticide image not found: {full_path}")
            return None
    except Exception as e:
        print(f"[ERROR] Error loading pesticide image {image_filename}: {e}")
        return None

def extract_pesticide_name(image_path):
    """Extract pesticide name from image filename"""
    try:
        filename = os.path.splitext(os.path.basename(image_path))[0]
        # Replace underscores and hyphens with spaces, title case
        name = filename.replace('-', ' ').replace('_', ' ').title()
        return name
    except Exception:
        return "Pesticide"

# --- Prediction Function ---
def predict_image(image_path):
    try:
        with Image.open(image_path) as img:
            image = img.convert("RGB")
            image = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error opening or processing image {image_path}: {e}")
        # Return a specific error structure if the image can't be read
        return {
            "prediction": "Processing Error",
            "is_pest": False, # Treat as not a pest for reporting
            "confidence": 0,
            "description": f"Could not read the image file: {os.path.basename(image_path)}.",
            "prevention": "Please ensure the file is a valid image.",
            "pesticides": "",
            "pesticide_images": []
        }


    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        max_prob, pred_class = torch.max(probs, 1)
        max_prob = max_prob.item()
        pred_class = pred_class.item()

    # Step 1: Pest check
    if max_prob < CONF_THRESHOLD:
        return {
            "prediction": "Not a Pest",
            "is_pest": False,
            "confidence": round(max_prob, 2),
            "description": "This image does not appear to contain a known pest.",
            "prevention": "No prevention needed.",
            "pesticides": "",
            "pesticide_images": []
        }

    # Step 2: Classify pest
    predicted_name = class_names[pred_class]
    info = pest_info_map.get(predicted_name, {})
    description = info.get("description", "No description available.")
    prevention = info.get("prevention", "No prevention info available.")
    pesticides = info.get("pesticides", "No pesticide info available.")
    
    # Get pesticide images
    pesticide_image_paths = info.get("pest_image", [])
    pesticide_images = []
    
    if isinstance(pesticide_image_paths, str):
        pesticide_image_paths = [pesticide_image_paths] if pesticide_image_paths else []
    
    for img_path in pesticide_image_paths:
        if img_path:  # Skip empty paths
            img_data_url = get_pesticide_image_data_url(img_path)
            if img_data_url:
                pesticide_name = extract_pesticide_name(img_path)
                pesticide_images.append({
                    "name": pesticide_name,
                    "image_url": img_data_url,
                    "path": img_path
                })

    return {
        "prediction": predicted_name,
        "class_name": predicted_name,
        "is_pest": True,
        "confidence": round(max_prob, 2),
        "description": description,
        "prevention": prevention,
        "pesticides": pesticides,
        "pesticide_images": pesticide_images
    }

# --- Flask App ---
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

@app.route("/")
def home():
    return render_template("Home.html")

@app.route("/login")
def login_page():
    return render_template("Login.html")

@app.route("/Description")
def description_page():
    return render_template("Description.html")

@app.route("/features")
def features_page():
    return render_template("Features.html")

@app.route("/about")
def about_page():
    return render_template("Contact.html")

@app.route("/upload")
def upload_page():
    return render_template("Upload.html")

@app.route("/methodology")
def methodology_page():
    return render_template("Methodology.html")

@app.route("/dataset")
def dataset_page():
    return render_template("Dataset.html")

@app.route("/image.png")
def serve_root_image():
    image_path = os.path.join(os.path.dirname(__file__), "image.png")
    return send_file(image_path, mimetype='image/png') if os.path.exists(image_path) else ("Image not found", 404)

@app.route("/architecture-1.png")
def serve_architecture_image():
    image_path = os.path.join(os.path.dirname(__file__), "architecture (1).png")
    return send_file(image_path, mimetype='image/png') if os.path.exists(image_path) else ("Image not found", 404)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    file.save(temp_file.name)
    temp_path = temp_file.name
    temp_file.close()

    try:
        result = predict_image(temp_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# --- Batch Prediction Endpoint ---
@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    results = []
    temp_files = []

    try:
        if "images" not in request.files:
            return "No image folder selected", 400
        
        files = request.files.getlist("images")
        for file in files:
            if file.filename and allowed_image(file.filename):
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
                file.save(temp_file.name)
                temp_files.append((file.filename, temp_file.name))

        if not temp_files:
            return "No valid image files found in the selected folder.", 400

        for filename, temp_path in temp_files:
            try:
                prediction_result = predict_image(temp_path)
                image_data_url = image_to_data_url(temp_path)
                
                status_class = "pest-detected" if prediction_result["is_pest"] else "not-pest"
                
                results.append({
                    "filename": filename,
                    "image_data_url": image_data_url,
                    "status": f"Pest: {prediction_result['prediction']}" if prediction_result["is_pest"] else "Not a Pest",
                    "status_class": status_class,
                    "confidence": prediction_result["confidence"],
                    "description": prediction_result["description"],
                    "pesticides": prediction_result.get("pesticides", ""),
                    "pesticide_images": prediction_result.get("pesticide_images", [])
                })
            except Exception as e:
                results.append({
                    "filename": filename, "image_data_url": None, "status": f"Error: {str(e)}",
                    "status_class": "error", "confidence": "N/A", "description": "Failed to process image",
                    "pesticide_images": []
                })
    finally:
        for _, temp_path in temp_files:
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except PermissionError:
                    print(f"Could not delete temp file (still in use): {temp_path}")
                except Exception as e:
                    print(f"Error deleting temp file {temp_path}: {e}")

    return generate_results_table(results)


def generate_results_table(results):
    """Generate HTML table with batch prediction results including pesticide images"""
    
    table_rows = []
    for i, result in enumerate(results, 1):
        if result["image_data_url"]:
            img_html = f'<img src="{result["image_data_url"]}" alt="{result["filename"]}" style="max-width: 120px; max-height: 120px; border-radius: 8px;">'
        else:
            img_html = '<span>No Preview</span>'
        
        status_color = {"pest-detected": "#dc2626", "not-pest": "#16a34a", "error": "#ea580c"}.get(result["status_class"], "#6b7280")
        
        pesticide_html = ""
        if result.get("pesticide_images"):
            pesticide_items = ""
            for p in result["pesticide_images"][:4]:
                pesticide_items += f'''
                    <div style="text-align: center; margin: 4px;">
                        <img src="{p["image_url"]}" alt="{p["name"]}" style="width: 60px; height: 60px; object-fit: cover; border-radius: 6px;">
                        <div style="font-size: 0.7rem; color: #374151;">{p["name"]}</div>
                    </div>
                '''
            pesticide_html = f'<div style="display: flex; flex-wrap: wrap; gap: 4px; justify-content: center;">{pesticide_items}</div>'
        
        table_rows.append(f"""
        <tr style="border-bottom: 1px solid #e5e7eb; vertical-align: top;">
            <td style="padding: 16px; text-align: center;">{i}</td>
            <td style="padding: 16px; text-align: center;">{img_html}</td>
            <td style="padding: 16px;">{result["filename"]}</td>
            <td style="padding: 16px; font-weight: 600; color: {status_color};">{result["status"]}</td>
            <td style="padding: 16px; text-align: center;">{result["confidence"]}</td>
            <td style="padding: 16px; max-width: 300px;">{result["description"]}</td>
            <td style="padding: 16px; min-width: 200px;">{pesticide_html if pesticide_html else '<span>N/A</span>'}</td>
        </tr>
        """)
    
    total = len(results)
    pests_detected = len([r for r in results if r['status_class'] == 'pest-detected'])
    clean_images = len([r for r in results if r['status_class'] == 'not-pest'])
    errors = len([r for r in results if r['status_class'] == 'error'])
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Batch Pest Detection Results</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f9fafb; margin: 20px; }}
            .container {{ max-width: 1600px; margin: 0 auto; background: white; border-radius: 16px; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1); overflow: hidden; }}
            .header {{ background: linear-gradient(135deg, #059669 0%, #047857 100%); color: white; padding: 30px; text-align: center; }}
            .summary {{ padding: 20px; background: #f8fafc; border-bottom: 1px solid #e5e7eb; display: flex; justify-content: space-around; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th {{ background: #f1f5f9; padding: 12px 16px; text-align: left; }}
            .actions {{ padding: 30px; text-align: center; background: #f8fafc; }}
            .btn {{ background: #059669; color: white; border: none; padding: 12px 24px; border-radius: 8px; text-decoration: none; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header"><h1>Batch Analysis Results</h1></div>
            <div class="summary">
                <div><strong>Total:</strong> {total}</div>
                <div><strong>Pests:</strong> {pests_detected}</div>
                <div><strong>Clean:</strong> {clean_images}</div>
                <div><strong>Errors:</strong> {errors}</div>
            </div>
            <table>
                <thead><tr>
                    <th>#</th><th>Preview</th><th>Filename</th><th>Result</th><th>Confidence</th><th>Description</th><th>Recommended Pesticides</th>
                </tr></thead>
                <tbody>{''.join(table_rows)}</tbody>
            </table>
            <div class="actions">
                <a href="/upload" class="btn">Analyze More</a>
                <a href="/" class="btn" style="margin-left: 10px;">Back to Home</a>
            </div>
        </div>
    </body>
    </html>
    """
    return html


if __name__ == "__main__":
    print("🚀 Server running at: http://127.0.0.1:5000")
    # Add a check for the pesticides folder on startup
    pesticides_folder = os.path.join(os.path.dirname(__file__), 'static', 'pesticides')
    if os.path.exists(pesticides_folder):
        files = os.listdir(pesticides_folder)
        print(f"✅ Found {len(files)} files in the pesticides folder: {pesticides_folder}")
    else:
        print(f"⚠️  Pesticides folder not found at: {pesticides_folder}")
        print("   Please create the folder and add pesticide images for them to appear.")
    
    # Get port from environment variable for production
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port, debug=False)