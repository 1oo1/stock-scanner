from datetime import datetime
from flask import Flask, render_template
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager
from app.config import get_flask_config
from app.utils.logger import get_logger

# Initialize logger
logger = get_logger()

# Initialize extensions
db = SQLAlchemy()
jwt = JWTManager()
migrate = Migrate()


def create_app():
    """Create and configure the Flask application"""

    app = Flask(__name__)

    # Load configuration
    app.config.from_object(get_flask_config())

    # Initialize extensions with app
    db.init_app(app)
    migrate.init_app(app, db)
    jwt.init_app(app)

    # Register blueprints
    from app.routes import all_blueprints

    for blueprint in all_blueprints:
        app.register_blueprint(blueprint)

    # Optional: Register error handlers
    @app.errorhandler(404)
    def not_found(error):
        # to page_not_found
        logger.error("访问了不存在的页面")
        return render_template("404.html"), 404

    @app.context_processor
    def inject_now():
        return {"now": datetime.utcnow()}

    return app
