from app import db
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import re


class User(db.Model):
    """User model for authentication"""

    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)

    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.set_password(password)

    def set_password(self, password):
        """Hash the password for security"""
        validation_result = self.validate_password(password)
        if validation_result is not True:
            raise ValueError(f"密码错误: {validation_result}")
        self.password_hash = generate_password_hash(password)

    def validate_password(self, password):
        """
        Validate password complexity:
        - At least 8 characters long
        - Contains at least one uppercase letter
        - Contains at least one lowercase letter
        - Contains at least one number
        - Contains at least one special character
        """
        if len(password) < 8:
            return "密码长度至少为8个字符"

        if not re.search(r"[A-Z]", password):
            return "至少包含一个大写字母"

        if not re.search(r"[a-z]", password):
            return "至少包含一个小写字母"

        if not re.search(r"[0-9]", password):
            return "至少包含一个数字"

        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return "至少包含一个特殊字符"

        return True

    def check_password(self, password):
        """Check if the provided password matches the hash"""
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        """String representation of the User object"""
        return f"<User {self.username}>"
