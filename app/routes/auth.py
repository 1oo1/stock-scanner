# Auth blueprint
from functools import wraps
from flask import (
    Blueprint,
    redirect,
    render_template,
    request,
    url_for,
    current_app,
)
from flask_jwt_extended import (
    create_access_token,
    jwt_required,
    unset_jwt_cookies,
    set_access_cookies,
    verify_jwt_in_request,
)

from app.models import User
from app import db
from app.utils.logger import get_logger

logger = get_logger()
auth_bp = Blueprint("auth", __name__)


def auth_or_login(f):
    """Decorator to check if the user is authenticated, if not, to login."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            # Check if the user is authenticated
            verify_jwt_in_request()
            return f(*args, **kwargs)
        except:
            # If not authenticated, redirect to login
            return redirect(url_for("auth.login"))

    return decorated_function


@auth_bp.route("/register", methods=["POST", "GET"])
def register():
    """Register route to create a new user."""
    if current_app.config["ENABLE_REGISTER_USER"] is False:
        return "{'msg': 'not found'}", 404

    if request.method == "GET":
        return render_template("register.html")

    username = request.form.get("username")
    email = request.form.get("email")
    password = request.form.get("password")
    invite_code = request.form.get("invite_code")

    if invite_code != current_app.config["REGISTER_PRIVATE_KEY"]:
        return render_template("register.html", error_msg="Invalid invite code")

    if User.query.filter_by(username=username).first():
        return render_template("register.html", error_msg="Username already exists")

    if User.query.filter_by(email=email).first():
        return render_template("register.html", error_msg="Email already exists")

    new_user = User(username=username, email=email, password=password)
    new_user.set_password(password)
    db.session.add(new_user)
    db.session.commit()

    return redirect(url_for("auth.login"))


@auth_bp.route("/login", methods=["POST", "GET"])
def login():
    """Login route to authenticate users and return JWT token."""

    # Check if already authenticated
    try:
        verify_jwt_in_request()
        return redirect(url_for("main.index"))
    except:
        pass

    if request.method == "GET":
        return render_template("login.html")

    username = request.form.get("username")
    password = request.form.get("password")

    user = User.query.filter_by(username=username).first()

    if user and user.check_password(password):
        access_token = create_access_token(identity=user.username)
        response = redirect(url_for("main.index"))
        set_access_cookies(response, access_token)
        logger.info(f"登录成功: {username}")
        return response
    else:
        logger.warning(f"登录失败: {username}")
        return render_template("login.html", error_msg="Bad username or password")


@auth_bp.route("/logout", methods=["POST"])
@jwt_required()
def logout():
    """Logout route to invalidate the JWT token."""
    # Invalidate the token by adding it to a blacklist (if implemented)
    # clear cookie
    response = redirect(url_for("auth.login"))
    unset_jwt_cookies(response)

    return response
