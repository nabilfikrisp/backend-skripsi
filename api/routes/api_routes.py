import torch

from flask import Blueprint, request
from api.controllers import api_controller
from PIL import Image
from pathlib import Path
from gnn_model.mobile_vig import mobilevig_ti
from torchvision.transforms import v2
from flask_cors import cross_origin

api_router = Blueprint("api", __name__, url_prefix="/api")

ALLOWED_EXTENSIONS = {"jpg", "jpeg"}

class_names = ["downdog", "goddess", "plank", "tree", "warrior2"]

model_path = Path("gnn_model/eksperimen_4_mobile_vig.pth")
checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
state_dict = checkpoint["model_state_dict"]
model = mobilevig_ti(num_classes=len(class_names))
model.load_state_dict(state_dict=state_dict)
model.to("cpu")
model.eval()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image):
    preprocess = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((224, 224)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image_tensor = preprocess(image)
    return image_tensor


def predict_image(image):
    imgt = preprocess_image(image).unsqueeze(dim=0).to("cpu")

    model.eval()
    with torch.inference_mode():
        target_image_pred = model(imgt)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    class_name = class_names[target_image_pred_label]
    probability = target_image_pred_probs.max().item()

    return {
        "code": 200,
        "message": "Success",
        "data": {"className": class_name, "probability": probability},
    }


@api_router.route("/", methods=["GET"])
def hello_router():
    return api_controller.hello_controller("hello world")


@api_router.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    try:
        if "file" not in request.files:
            return {"code": 400, "message": "No file part"}

        file = request.files["file"]

        if file.filename == "":
            return {"code": 400, "message": "No selected file"}

        if not allowed_file(file.filename):
            return {"code": 400, "message": "Only jpg and jpeg image is accepted"}

        if file and allowed_file(file.filename):
            img = Image.open(file)
            return predict_image(img)

    except Exception as e:
        return {"code": 500, "message": f"An error occurred: {str(e)}"}
