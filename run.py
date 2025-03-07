from app import create_app

app = create_app()

if __name__ == "__main__":
    config = app.config
    app.run(
        host=config.get("FLASK_HOST", "127.0.0.1"), port=config.get("FLASK_PORT", 8888)
    )
