import os
import json
import torch
import torch.nn as nn
import timm
import torchvision.transforms as T
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

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

# --- Class Names ---
class_names = [
    "Dolycoris baccarum", "Lycorma delicatula", "Eurydema dominulus", "Pieris rapae",
    "Halyomorpha halys", "Spilosoma obliqua", "Graphosoma rubrolineata", "Luperomorpha suturalis",
    "Leptocorisa acuta", "Sesamia inferens", "Cicadella viridis", "Callitettix versicolor",
    "Scotinophara lurida", "Cletus punctiger", "Nezara viridula", "Dicladispa armigera",
    "Riptortus pedestris", "Maruca testulalis", "Chauliops fallax", "Chilo suppressalis",
    "Stollia ventralis", "Nilaparvata lugens", "Diostrombus politus", "Phyllotreta striolata",
    "Aulacophora indica", "Laodelphax striatellus", "Ceroplastes ceriferus", "Corythucha marmorata",
    "Dryocosmus kuriphilus", "Porthesia taiwana", "Chromatomyia horticola", "Iscadia inexacta",
    "Plutella xylostella", "Empoasca flavescens", "Dolerus tritici", "Spodoptera litura",
    "Corythucha ciliata", "Bemisia tabaci", "Ceutorhynchus asper", "Strongyllodes variegatus"
]

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
