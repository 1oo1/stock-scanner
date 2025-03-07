from app import create_app

app = create_app()

if __name__ == "__main__":
    config = app.config
    app.run(host=config.get("HOST", "127.0.0.1"), port=config.get("PORT", 8888))
