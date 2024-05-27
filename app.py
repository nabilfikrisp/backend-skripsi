# app.py
from flask import Flask
from api.routes.api_routes import api_router

app = Flask(__name__)
app.register_blueprint(api_router)

app.config["UPLOAD_EXTENSIONS"] = [".jpg", ".jpeg"]
app.config["MAX_CONTENT_LENGTH"] = 16 * 1000 * 1000


@app.route("/")
def hello_world():
    return {"message": "hello, please access /api"}
