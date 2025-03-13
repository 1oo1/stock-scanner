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
    unset_jwt_cookies,
    set_access_cookies,
    verify_jwt_in_request,
    get_jwt_identity,
)

from app.models.users import User
from app import db
from app.utils.logger import get_logger

logger = get_logger()
auth_bp = Blueprint("auth", __name__)


def auth_or_login(f):
    """Decorator to check if the user is authenticated, if not, redirect to login."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            # Check if the user is authenticated
            verify_jwt_in_request()
            return f(*args, **kwargs)
        except Exception as e:
            # logger.error(f"Auth error details: {type(e).__name__}: {str(e)}")
            # If not authenticated, redirect to login with the current URL
            return redirect(url_for("auth.login", next=request.url))

    return decorated_function


@auth_bp.route("/register", methods=["POST", "GET"])
def register():
    """Register route to create a new user."""
    if current_app.config["ENABLE_REGISTER_USER"] is False:
        return render_template("register.html", disabled_register=True)

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
        return redirect(url_for("pages.index"))
    except:
        pass

    if request.method == "GET":
        return render_template("login.html")

    username = request.form.get("username")
    password = request.form.get("password")
    next_url = request.form.get("next")

    user = User.query.filter_by(username=username).first()

    if user and user.check_password(password):
        access_token = create_access_token(identity=user.username)
        # Get the next URL if it exists, otherwise default to pages.index
        response = redirect(next_url or url_for("pages.index"))
        set_access_cookies(response, access_token)
        logger.info(f"登录成功: {username}")
        return response
    else:
        logger.warning(f"登录失败: {username}")
        return render_template("login.html", error_msg="Bad username or password")


@auth_bp.route("/logout", methods=["GET"])
@auth_or_login
def logout():
    """Logout route to invalidate the JWT token."""
    # Invalidate the token by adding it to a blacklist (if implemented)
    # clear cookie
    response = redirect(url_for("pages.index"))
    unset_jwt_cookies(response)

    logger.info("用户登出")
    return response


@auth_bp.route("/profile", methods=["GET", "POST"])
@auth_or_login
def profile():
    """Profile page to view and update user information."""
    username = get_jwt_identity()
    user = User.query.filter_by(username=username).first()

    if request.method == "POST":
        # Handle JSON request from API
        data = request.get_json()
        current_password = data.get("current_password")
        new_password = data.get("new_password")
        confirm_password = data.get("confirm_password")

        if not user.check_password(current_password):
            return {"success": False, "error": "当前密码不正确"}, 400

        if new_password != confirm_password:
            return {"success": False, "error": "两次输入的新密码不匹配"}, 400

        try:
            user.set_password(new_password)
            db.session.commit()
            logger.info(f"密码已更改: {username}")
        except ValueError as e:
            # Handle password validation errors
            return {"success": False, "error": str(e)}, 400
        except Exception as e:
            logger.error(f"密码更新失败: {username}, 错误: {str(e)}")
            return {"success": False, "error": "密码更新失败"}, 500

        return {"success": True, "message": "密码更新成功"}, 200

    return render_template("profile.html", user=user)
