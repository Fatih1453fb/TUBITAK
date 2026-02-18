from app import create_app
from modules.ml_model import compare_model_variants

app = create_app()

if __name__ == "__main__":
    with app.app_context():
        # Compare for Building 1
        compare_model_variants(1)
