# API routes/blueprints
from flask import Flask


def init_app_routes(app: Flask):
    # Import blueprints
    from app.routes.apis import apis_bp
    from app.routes.pages import pages_bp
    from app.routes.auth import auth_bp

    # Register blueprints with the app
    app.register_blueprint(apis_bp)
    app.register_blueprint(pages_bp)
    app.register_blueprint(auth_bp)
