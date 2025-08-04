from flask import Flask
from app.config import Config
from .features.core import bp_core_web, bp_core_api


def create_app():
    # initialize app
    app = Flask(__name__)

    # configure application
    app.config.from_object(Config)

    # initialize extensions
    ...

    # register blueprints
    app.register_blueprint(bp_core_web)
    app.register_blueprint(bp_core_api)

    return app
