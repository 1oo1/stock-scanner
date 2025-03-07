from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager
from app.config import get_flask_config
from app.utils.logger import get_logger

# Initialize logger
logger = get_logger()

# Initialize extensions
db = SQLAlchemy()
jwt = JWTManager()


def create_app():
    """Create and configure the Flask application"""

    app = Flask(__name__)

    # Load configuration
    app.config.from_object(get_flask_config())

    # Initialize extensions with app
    # db.init_app(app)
    jwt.init_app(app)

    # Register blueprints
    from app.routes import all_blueprints

    for blueprint in all_blueprints:
        app.register_blueprint(blueprint)

    # Optional: Register error handlers
    @app.errorhandler(404)
    def not_found(error):
        return {"error": "Not found"}, 404

    return app
