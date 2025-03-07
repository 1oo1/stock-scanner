# Auth blueprint
from functools import wraps
from flask import Blueprint, redirect, render_template, request, jsonify, url_for
from flask_jwt_extended import create_access_token, jwt_required

from app.models import User
from app import db


auth_bp = Blueprint("auth", __name__)


def auth_or_login(f):
    """Decorator to check if the user is authenticated, if not, to login."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            # Check if the user is authenticated
            jwt_required()(f)(*args, **kwargs)
        except Exception:
            # If not authenticated, redirect to login
            return redirect(url_for("auth.login"))

    return decorated_function


@auth_bp.route("/login", methods=["POST", "GET"])
def login():
    """Login route to authenticate users and return JWT token."""

    if request.method == "GET":
        return render_template("login.html")

    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    user = User.query.filter_by(username=username).first()

    if user and user.check_password(password):
        access_token = create_access_token(identity={"username": user.username})
        return jsonify(access_token=access_token), 200
    else:
        return jsonify({"msg": "Bad username or password"}), 401


@auth_bp.route("/logout", methods=["POST"])
@jwt_required()
def logout():
    """Logout route to invalidate the JWT token."""
    # Invalidate the token by adding it to a blacklist (if implemented)
    # or simply not returning it in the response.
    return jsonify({"msg": "Logout successful"}), 200
