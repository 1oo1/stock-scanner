from flask import Blueprint, render_template
from app.routes.auth import auth_or_login

# Create a blueprint for the page routes
pages_bp = Blueprint("pages", __name__)


@pages_bp.route("/")
def index():
    return render_template("index.html")


@pages_bp.route("/tech_analysis")
@auth_or_login
def tech_analysis():
    return render_template("tech_analysis.html")
