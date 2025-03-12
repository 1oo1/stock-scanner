from flask import Flask, render_template
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, get_jwt_identity, verify_jwt_in_request
from app.config import get_flask_config
from app.utils.logger import get_logger
from app.routes import init_app_routes

app = Flask(__name__)

# Initialize logger
logger = get_logger()

# Initialize extensions
db = SQLAlchemy()
jwt = JWTManager()
migrate = Migrate()


# Optional: Register error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template("404.html"), 404


@app.context_processor
def inject_global():
    current_user = {"is_authenticated": False, "username": None}
    try:
        verify_jwt_in_request(optional=True)
        user_name = get_jwt_identity()
        if user_name:
            current_user["is_authenticated"] = True
            current_user["username"] = user_name
    except Exception as e:
        pass

    return {"current_user": current_user}


def create_app():
    """Create and configure the Flask application"""
    # Load configuration
    app.config.from_object(get_flask_config())

    # Initialize extensions with app
    db.init_app(app)
    migrate.init_app(app, db)
    jwt.init_app(app)

    # Register blueprints
    init_app_routes(app)

    return app
