# app.py
    from flask import Flask
    from api.routes.api_routes import api_router
    from flask_cors import CORS

    app = Flask(__name__)
    cors = CORS(app)
    app.register_blueprint(api_router)

@app.route("/")
def hello_world():
    return {"message": "hello, please access /api"}
