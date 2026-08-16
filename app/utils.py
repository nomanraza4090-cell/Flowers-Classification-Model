from PIL import Image

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}

def valid_image(file):
    try:
        Image.open(file).verify()
        return True
    except Exception:
        return False
