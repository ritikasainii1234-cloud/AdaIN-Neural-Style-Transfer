import os
import uuid
from pathlib import Path

import torch
from flask import Flask, render_template, request, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from PIL import Image
from torchvision import transforms

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


# -------------------------------
# 1. PATH CONFIGURATION
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
VGG_PATH = BASE_DIR / "vgg_normalised.pth"
DECODER_PATH = BASE_DIR / "experiment" / "final_exp" / "decoder_final.pth"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


# -------------------------------
# 2. FLASK APPLICATION
# -------------------------------

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "adain-local-demo-secret")
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

Bootstrap(app)


# -------------------------------
# 3. UPLOAD FORM
# -------------------------------

class UploadForm(FlaskForm):
    content = FileField("Content Image")
    style = FileField("Style Image")
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField("Alpha", default=1.0)
    submit = SubmitField("Transfer Style")


# -------------------------------
# 4. LOAD PRETRAINED MODELS
# -------------------------------

device = torch.device("cpu")

print("Using device:", device)

print("Loading VGG Encoder...")
encoder = VGGEncoder(str(VGG_PATH)).to(device)

print("Loading AdaIN Decoder...")
decoder = Decoder().to(device)

decoder.load_state_dict(
    torch.load(
        str(DECODER_PATH),
        map_location=device,
        weights_only=True
    )
)

encoder.eval()
decoder.eval()

print("Models loaded successfully!")


# -------------------------------
# 5. HELPER FUNCTIONS
# -------------------------------

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in app.config["ALLOWED_EXTENSIONS"]
    )


def save_uploaded_file(file):
    if not file or not file.filename:
        return None

    if not allowed_file(file.filename):
        return None

    original_name = secure_filename(file.filename)

    unique_name = f"{uuid.uuid4().hex}_{original_name}"

    save_path = UPLOAD_FOLDER / unique_name

    file.save(str(save_path))

    return unique_name


# -------------------------------
# 6. ADAIN STYLE TRANSFER
# -------------------------------

def style_transfer(content_image, style_image, alpha=1.0):

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    content_tensor = transform(content_image).unsqueeze(0).to(device)
    style_tensor = transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():

        content_features = encoder(
            content_tensor,
            is_test=True
        )

        style_features = encoder(
            style_tensor,
            is_test=True
        )

        stylized_features = adaptive_instance_normalization(
            content_features,
            style_features
        )

        stylized_features = (
            alpha * stylized_features
            + (1 - alpha) * content_features
        )

        output = decoder(stylized_features)

    return output


# -------------------------------
# 7. SAVE GENERATED IMAGE
# -------------------------------

def save_output_image(image, filename):

    image = image.detach().cpu()
    image = image.squeeze(0)
    image = image.clamp(0, 1)

    pil_image = transforms.ToPILImage()(image)

    output_path = UPLOAD_FOLDER / filename

    pil_image.save(str(output_path))


# -------------------------------
# 8. MAIN WEBSITE ROUTE
# -------------------------------

@app.route("/", methods=["GET", "POST"])
def index():

    form = UploadForm()

    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if request.method == "POST":

        if form.validate_on_submit():

            content_file = form.content.data
            style_file = form.style.data

            content_filename = save_uploaded_file(content_file)
            style_filename = save_uploaded_file(style_file)

            if not content_filename:
                previous_content = form.content_path.data
                if previous_content:
                    if (UPLOAD_FOLDER / previous_content).is_file():
                        content_filename = previous_content

            if not style_filename:
                previous_style = form.style_path.data
                if previous_style:
                    if (UPLOAD_FOLDER / previous_style).is_file():
                        style_filename = previous_style

            if not content_filename:
                error = "Please upload a valid content image."

            elif not style_filename:
                error = "Please upload a valid style image."

            else:

                try:

                    alpha = form.alpha.data

                    if alpha is None or not (0.0 <= alpha <= 1.0):
                        raise ValueError(
                            "Alpha must be between 0 and 1."
                        )

                    content_path = UPLOAD_FOLDER / content_filename
                    style_path = UPLOAD_FOLDER / style_filename

                    content_image = Image.open(
                        content_path
                    ).convert("RGB")

                    style_image = Image.open(
                        style_path
                    ).convert("RGB")

                    stylized_image = style_transfer(
                        content_image,
                        style_image,
                        alpha
                    )

                    result_filename = (
                        f"stylized_{uuid.uuid4().hex}.png"
                    )

                    save_output_image(
                        stylized_image,
                        result_filename
                    )

                    result_image = result_filename

                except Exception as e:
                    error = str(e)

        else:
            error = "Please check the uploaded images and form values."

    return render_template(
        "index.html",
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error
    )


# -------------------------------
# 9. IMAGE ROUTES
# -------------------------------

@app.route("/uploads/<filename>")
def send_image(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


@app.route("/examples/<path:filename>")
def send_example(filename):
    examples_folder = BASE_DIR.parent / "Demo_IO_Images"

    print("Looking for image:", examples_folder / filename)

    return send_from_directory(
        str(examples_folder),
        filename
    )


# -------------------------------
# 10. RUN FLASK WEBSITE
# -------------------------------

if __name__ == "__main__":

    print("\nStarting AdaIN Neural Style Transfer Website...")
    print("Open: http://127.0.0.1:5000")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )

