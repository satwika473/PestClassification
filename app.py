import os
from dotenv import load_dotenv
import json
import torch
import torch.nn as nn
import timm
import torchvision.transforms as T
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS

# Load environment variables from .env file
load_dotenv()

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
# The model was trained with ImageFolder which sorts folders alphabetically: '1', '10', '11', etc.
# We need to create the correct mapping from model output index to pest name

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
# ImageFolder sorts: '1', '10', '11', '12', ..., '19', '2', '20', ..., '29', '3', ..., '9'
folder_names_sorted = sorted([str(i) for i in range(1, 41)], key=str)  # Sort as strings
class_names = [label_to_name[int(folder_name)] for folder_name in folder_names_sorted]

# --- Load Pest Details from JSON (convert list → dict) ---
pest_details_path = os.path.join(os.path.dirname(__file__), "pest.pest_details.json")
if not os.path.exists(pest_details_path):
    raise FileNotFoundError(f"Pest details file not found: {pest_details_path}")

with open(pest_details_path, "r", encoding="utf-8") as f:
    pest_list = json.load(f)

# Convert to dictionary keyed by pest name
pest_info_map = {
    pest["name"]: {
        "description": pest.get("description", "No description available."),
        "prevention": pest.get("prevention", "No prevention info available.")
    }
    for pest in pest_list
}

# --- Confidence Threshold ---
CONF_THRESHOLD = 0.5  # adjust after testing

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Model ---
model = InsectModel(num_classes=len(class_names))
model_path = os.path.join(os.path.dirname(__file__), "vit_best.pth")

def download_file_from_google_drive(file_id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    print(f"Downloading model file from Google Drive (id: {file_id})...")
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)
    print("Model downloaded successfully!")

if not os.path.exists(model_path):
    MODEL_URL = os.environ.get('MODEL_URL')
    if not MODEL_URL:
        raise ValueError("MODEL_URL environment variable is not set")

    # Extract file ID from Google Drive URL
    if 'drive.google.com' in MODEL_URL:
        try:
            parsed = urlparse(MODEL_URL)
            if 'id' in parse_qs(parsed.query):
                file_id = parse_qs(parsed.query)['id'][0]
            else:
                file_id = parsed.path.split('/')[-2]
            download_file_from_google_drive(file_id, model_path)
        except Exception as e:
            raise Exception(f"Failed to download model from Google Drive: {str(e)}")
    else:
        raise ValueError("MODEL_URL must be a Google Drive link")

try:
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
except Exception as e:
    if os.path.exists(model_path):
        os.remove(model_path)  # Remove potentially corrupted file
    raise Exception(f"Error loading model: {str(e)}")

# --- Image Preprocessing ---
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

# --- Prediction Function ---
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

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
            "prevention": "No prevention needed."
        }

    # Step 2: Classify pest
    predicted_name = class_names[pred_class]
    info = pest_info_map.get(predicted_name, {})
    description = info.get("description", "No description available.")
    prevention = info.get("prevention", "No prevention info available.")

    return {
        "prediction": predicted_name,
        "class_name": predicted_name,
        "is_pest": True,
        "confidence": round(max_prob, 2),
        "description": description,
        "prevention": prevention
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
    """Serve the image.png file from root directory"""
    image_path = os.path.join(os.path.dirname(__file__), "image.png")
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/png')
    else:
        return "Image not found", 404
@app.route("/architecture-1.png")
def serve_architecture_image():
    """Serve the architecture (1).png file from root directory"""
    image_path = os.path.join(os.path.dirname(__file__), "architecture (1).png")
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/png')
    else:
        return "Image not found", 404
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    temp_path = os.path.join("temp.jpg")
    file.save(temp_path)

    try:
        result = predict_image(temp_path)
        os.remove(temp_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Server running at: http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
