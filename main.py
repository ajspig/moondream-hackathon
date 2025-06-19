import moondream as md
import os
from dotenv import load_dotenv
from io import BytesIO
import requests
from IPython.display import display
from PIL import Image, ImageDraw

load_dotenv() # Load variables from .env


def get_nyctmc_camera_image(camera_id, save_path=None):
    """
    Get image from NYCTMC API using camera ID
    
    Args:
        camera_id (str): Camera ID (e.g., 'eafc65f5-6ff9-4203-905f-3995b9fbc9eb')
        save_path (str): Optional path to save the image
        
    Returns:
        PIL.Image or None: Image object or None if failed
    """
    try:
        image_url = f"https://nyctmc.org/api/cameras/{camera_id}/image"
        
        response = requests.Session().get(image_url, timeout=10)
        response.raise_for_status()
        
        image = Image.open(BytesIO(response.content))
        if save_path:
            os.makedirs("images", exist_ok=True)
            save_path = os.path.join("images", save_path, f"{camera_id}.jpg")
            image.save(save_path)
            print(f"Image saved to: {save_path}")
        
        return image
    except requests.RequestException as e:
        print(f"Error fetching camera {camera_id}: {e}")
        return None
    except Exception as e:
        print(f"Error processing image for camera {camera_id}: {e}")
        return None

def main():

    # This will run the model locally
    # model = md.vl(endpoint="http://localhost:2020/v1")

    model = md.vl(api_key=os.getenv("MOONDREAM_API_KEY"))
    # https://nyctmc.org/api/cameras/eafc65f5-6ff9-4203-905f-3995b9fbc9eb/image
    image = get_nyctmc_camera_image("eafc65f5-6ff9-4203-905f-3995b9fbc9eb",  save_path="nyctmc")
    if image:
        # Generate a caption for the camera image
        caption_response = model.caption(image, length="short")
        print(caption_response["caption"])

        answer = model.query(image, "Are there cars in the image?")['answer']
        print(f"Answer to query: {answer}")
        
        detection = model.detect(image, 'car')['objects']
        points = model.point(image, 'car')['points']
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        w, h = overlay.size
        for box in detection:
            draw.rectangle([
                int(box['x_min'] * w),
                int(box['y_min'] * h),
                int(box['x_max'] * w),
                int(box['y_max'] * h)
            ], outline='red', width=3)
        for pt in points:
            r = 4
            draw.ellipse([
                int(pt['x'] * w) - r, int(pt['y'] * h) - r,
                int(pt['x'] * w) + r, int(pt['y'] * h) + r
            ], fill='blue')
        output_path = "images/nyctmc_with_detections.jpg"
        overlay.save(output_path)

    else:
        print("Failed to retrieve camera image.")


if __name__ == "__main__":
    main()
