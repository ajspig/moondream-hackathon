import moondream as md
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv() # Load variables from .env

def main():

    # This will run the model locally
    # model = md.vl(endpoint="http://localhost:2020/v1")

    model = md.vl(api_key=os.getenv("MOONDREAM_API_KEY"))
    image = Image.open("images/frieren.jpg")

    # Example: Generate a caption
    caption_response = model.caption(image, length="short")
    print(caption_response["caption"])


if __name__ == "__main__":
    main()
