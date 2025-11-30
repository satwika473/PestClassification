import os
import json
import torch
import torch.nn as nn
import timm
import torchvision.transforms as T
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import io
import base64
import tempfile

class InsectModel(nn.Module):
    def __init__(self, num_classes):
        super(InsectModel, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)

    def forward(self, image):
        return self.model(image)

# Load data efficiently
def load_app_data():
    base_dir = os.path.dirname(__file__)
    
    # Load class mapping
    labeled_classes_path = os.path.join(base_dir, "labeled_classes.txt")
    if not os.path.exists(labeled_classes_path):
        raise FileNotFoundError(f"labeled_classes.txt not found: {labeled_classes_path}")
    
    label_to_name = {}
    with open(labeled_classes_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                label_to_name[int(parts[0])] = parts[1]
    
    folder_names_sorted = sorted([str(i) for i in range(1, 41)], key=str)
    class_names = [label_to_name.get(int(folder_name), f"Unknown_{folder_name}") for folder_name in folder_names_sorted]
    
    # Load pest details
    pest_details_path = os.path.join(base_dir, "pest.pest_details.json")
    if not os.path.exists(pest_details_path):
        raise FileNotFoundError(f"Pest details file not found: {pest_details_path}")
    
    with open(pest_details_path, "r", encoding="utf-8") as f:
        pest_list = json.load(f)
    
    pest_info_map = {}
    for pest in pest_list:
        pest_name = pest.get("name", "")
        if pest_name:
            pest_info_map[pest_name] = {
                "description": pest.get("description", "No description available."),
                "prevention": pest.get("prevention", "No prevention info available."),
                "pesticides": pest.get("pesticides", "No pesticide info available."),
                "pest_image": pest.get("pest_image", [])
            }
    
    return class_names, pest_info_map

# Initialize data
class_names, pest_info_map = load_app_data()

# Load model with optimizations
def load_optimized_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InsectModel(num_classes=len(class_names))
    
    model_path = os.path.join(os.path.dirname(__file__), "vit_best.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load model state dict only (no optimizer states)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, device

model, device = load_optimized_model()
transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
CONF_THRESHOLD = 0.5

# Helper functions
def get_pesticide_image_data_url(image_filename, max_size=200):
    try:
        base_dir = os.path.dirname(__file__)
        safe_filename = os.path.basename(image_filename)
        full_path = os.path.join(base_dir, 'static', 'pesticides', safe_filename)
        
        if os.path.exists(full_path):
            with Image.open(full_path) as img:
                img = img.convert("RGB")
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                img_data = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/jpeg;base64,{img_data}"
        return None
    except Exception as e:
        print(f"Error loading pesticide image {image_filename}: {e}")
        return None

def extract_pesticide_name(image_path):
    try:
        filename = os.path.splitext(os.path.basename(image_path))[0]
        return filename.replace('-', ' ').replace('_', ' ').title()
    except:
        return "Pesticide"

def predict_image(image_path):
    try:
        with Image.open(image_path) as img:
            image = transform(img.convert("RGB")).unsqueeze(0).to(device)
    except Exception as e:
        return {
            "prediction": "Processing Error",
            "is_pest": False,
            "confidence": 0,
            "description": f"Could not read image: {str(e)}",
            "prevention": "",
            "pesticides": "",
            "pesticide_images": []
        }

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        max_prob, pred_class = torch.max(probs, 1)
        max_prob, pred_class = max_prob.item(), pred_class.item()

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

    predicted_name = class_names[pred_class]
    info = pest_info_map.get(predicted_name, {})
    
    # Get pesticide images
    pesticide_image_paths = info.get("pest_image", [])
    pesticide_images = []
    
    if isinstance(pesticide_image_paths, str):
        pesticide_image_paths = [pesticide_image_paths] if pesticide_image_paths else []
    
    for img_path in pesticide_image_paths:
        if img_path:
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
        "is_pest": True,
        "confidence": round(max_prob, 2),
        "description": info.get("description", "No description available."),
        "prevention": info.get("prevention", "No prevention info available."),
        "pesticides": info.get("pesticides", "No pesticide info available."),
        "pesticide_images": pesticide_images
    }

# Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Essential routes only
@app.route("/")
def home():
    return render_template("Home.html")

@app.route("/upload")
def upload_page():
    return render_template("Upload.html")

@app.route("/Description")
def description_page():
    return render_template("Description.html")

@app.route("/about")
def about_page():
    return render_template("Contact.html")

@app.route("/methodology")
def methodology_page():
    return render_template("Methodology.html")

@app.route("/dataset")
def dataset_page():
    return render_template("Dataset.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        file.save(temp_file.name)
        temp_path = temp_file.name

    try:
        result = predict_image(temp_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

# Simplified batch processing
@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    if "images" not in request.files:
        return "No images selected", 400
    
    files = request.files.getlist("images")
    results = []
    
    for file in files:
        if file.filename and file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                file.save(temp_file.name)
                
                try:
                    result = predict_image(temp_file.name)
                    status = f"Pest: {result['prediction']}" if result["is_pest"] else "Not a Pest"
                    results.append(f"<tr><td>{file.filename}</td><td>{status}</td><td>{result['confidence']}</td></tr>")
                except Exception as e:
                    results.append(f"<tr><td>{file.filename}</td><td>Error: {str(e)}</td><td>N/A</td></tr>")
                finally:
                    try:
                        os.remove(temp_file.name)
                    except:
                        pass
    
    # Simple HTML response
    html = f"""
    <!DOCTYPE html>
    <html><head><title>Batch Results</title></head>
    <body style="font-family: Arial, sans-serif; margin: 40px;">
        <h1>Batch Analysis Results</h1>
        <table border="1" style="border-collapse: collapse; width: 100%;">
            <tr style="background: #f0f0f0;"><th>Filename</th><th>Result</th><th>Confidence</th></tr>
            {''.join(results)}
        </table>
        <br><a href="/upload" style="background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Analyze More</a>
    </body></html>
    """
    return html

if __name__ == "__main__":
    print("🚀 Starting optimized server...")
    
    # Check pesticides folder
    pesticides_folder = os.path.join(os.path.dirname(__file__), 'static', 'pesticides')
    if os.path.exists(pesticides_folder):
        files = os.listdir(pesticides_folder)
        print(f"✅ Found {len(files)} pesticide images")
    else:
        print("⚠️ Pesticides folder not found")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port, debug=False)