# API routes/blueprints
from app.routes.auth import auth_bp
from app.routes.main import main_bp

# List of all blueprints for easier registration
all_blueprints = [auth_bp, main_bp]
