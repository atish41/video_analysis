import os
import tempfile
import logging
from io import BytesIO
from flask import Flask, request, jsonify
import requests
from PIL import Image
import cv2
import numpy as np
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import re
from urllib.parse import urlparse
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Initialize Vertex AI
vertexai.init(
    project=os.getenv('VERTEX_PROJECT'), 
    location=os.getenv('VERTEX_LOCATION')
)

system_instruction = """
You are Paddi AI, an AI travel assistant. 
When presented with an image or video, you will:
1. Analyze the visual content thoroughly
2. Provide a detailed, descriptive analysis of what you observe
3. If a text message is provided along with the media, incorporate that context into your analysis
4. Answer any specific questions about the media if they are present
5. Focus on details relevant to travel, destinations, cultural insights, or interesting visual elements

Your response should be clear, comprehensive, and informative, highlighting the most notable aspects of the image or video.
"""

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 0.2, 
    "top_p": 0.95,
}

model = GenerativeModel(model_name="gemini-1.5-flash-001", system_instruction=[system_instruction])

# def fetch_and_preprocess_image(image_path):
#     """
#     Fetch and preprocess an image from a given path or URL
    
#     Args:
#         image_path (str): Path or URL of the image
    
#     Returns:
#         tuple: Processed image part for Vertex AI and original image
#     """
#     headers = {
#         "User-Agent": (
#             "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
#             "Chrome/85.0.4183.121 Safari/537.36"
#         )
#     }
    
#     try:
#         if image_path.startswith(("http://", "https://")):
#             try:
#                 response = requests.get(image_path, stream=True, headers=headers, verify=True)
#                 response.raise_for_status()
#                 image = Image.open(BytesIO(response.content))
#             except requests.RequestException as e:
#                 logging.error(f"Error fetching image {image_path}: {e}")
#                 raise
#         else:
#             image = Image.open(image_path)

#         # Convert RGBA to RGB if necessary
#         if image.mode == 'RGBA':
#             image = image.convert('RGB')

#         # Resize and process the image
#         processed_image = image.resize((224, 224), Image.Resampling.LANCZOS)
#         buffer = BytesIO()
#         processed_image.save(buffer, format="JPEG")
#         image_bytes = buffer.getvalue()
#         return Part.from_data(mime_type="image/jpeg", data=image_bytes), image

#     except Exception as e:
#         logging.error(f"Error processing image: {e}")
#         raise


def fetch_and_preprocess_image(image_path):
    headers = {
        "User-Agent": "Python Image Fetcher",
    }
    
    try:
        if image_path.startswith(("http://", "https://")):
            try:
                # Disable SSL verification if needed
                response = requests.get(image_path, stream=True, headers=headers, verify=False)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            except requests.RequestException as e:
                logging.error(f"Detailed error fetching image {image_path}: {e}")
                # Log full traceback
                import traceback
                logging.error(traceback.format_exc())
                raise
        else:
            image = Image.open(image_path)

        # Convert RGBA to RGB if necessary
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # Resize and process the image
        processed_image = image.resize((224, 224), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        processed_image.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()
        return Part.from_data(mime_type="image/jpeg", data=image_bytes), image

    except Exception as e:
        logging.error(f"Error processing image: {e}")
        raise

    
def fetch_and_preprocess_video(video_path):
    """
    Fetch and preprocess a video from a given path or URL
    
    Args:
        video_path (str): Path or URL of the video
    
    Returns:
        tuple: List of video frame parts and first frame
    """
    temp_file_path = None
    supported_formats = ['.mp4', '.mov', '.avi', '.mkv', '.3gp', '.flv']

    try:
        if video_path.startswith(("http://", "https://")):
            response = requests.get(video_path, stream=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mkv") as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                temp_file_path = temp_file.name
        else:
            temp_file_path = video_path

        if not any(temp_file_path.endswith(ext) for ext in supported_formats):
            logging.error(f"Unsupported video format: {video_path}")
            raise ValueError(f"Unsupported video format: {video_path}")

        cap = cv2.VideoCapture(temp_file_path)
        if not cap.isOpened():
            logging.error(f"Failed to open video file: {video_path}")
            raise ValueError(f"Failed to open video file: {video_path}")

        frames = []
        success, frame = cap.read()
        frame_interval = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 5)

        # Convert first frame to PIL Image for returning
        first_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        while success and len(frames) < 5:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame)
            pil_image = pil_image.resize((224, 224), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            pil_image.save(buffer, format="JPEG")
            frames.append(Part.from_data(mime_type="image/jpeg", data=buffer.getvalue()))
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + frame_interval)
            success, frame = cap.read()

        cap.release()
        return frames, first_frame

    except Exception as e:
        logging.error(f"Error processing video: {e}")
        raise
    finally:
        if temp_file_path and temp_file_path != video_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def analyze_media(media_type, media_url, text_msg=""):
    """
    Analyze media content using Vertex AI
    
    Args:
        media_type (str): Type of media ('video' or 'image')
        media_url (str): URL of the media
        text_msg (str, optional): Additional text message. Defaults to "".
        
    Returns:
        dict: Analysis details including description and image details
    """
    # Prepare content with optional text message
    contents = [Part.from_text(text_msg)] if text_msg else [Part.from_text("No additional context")]

    try:
        if media_type == 'video':
            # For video, take the first frame or all frames
            video_frames, first_frame = fetch_and_preprocess_video(media_url)
            content = contents + [video_frames[0]]  # Prioritize the first frame
            
            response = model.generate_content(content, generation_config=generation_config)
            
            # Use tempfile to create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                first_frame.save(temp_file.name)
                first_frame_path = temp_file.name
            
            return {
                "description": response.text.strip(),
                "media_type": "video",
                "media_path": first_frame_path
            }
        
        elif media_type == 'image':
            image_part, original_image = fetch_and_preprocess_image(media_url)
            content = contents + [image_part]
            
            response = model.generate_content(content, generation_config=generation_config)
            
            # Use tempfile to create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                original_image.save(temp_file.name)
                image_path = temp_file.name
            
            return {
                "description": response.text.strip(),
                "media_type": "image",
                "media_path": image_path
            }
        
        else:
            raise ValueError("Invalid media type. Must be 'video' or 'image'.")

    except Exception as e:
        logging.error(f"Error analyzing media: {e}")
        raise

app = Flask(__name__)

# @app.route('/analyze', methods=['POST'])
# def analyze():
#     """
#     Endpoint for media analysis
#     """
#     data = request.get_json()
    
#     # Validate request structure
#     if 'type' not in data or 'media' not in data:
#         return jsonify({"error": "Invalid request structure. Requires 'type' and 'media'."}), 400
    
#     media_type = data['type']
#     media_url = data['media']
#     text_msg = data.get('text_msg', '')
    
#     # Validate media type
#     if media_type not in ['video', 'image']:
#         return jsonify({"error": "Media type must be 'video' or 'image'."}), 400
    
#     # Prioritize video
#     if media_type == 'image' and 'video' in data:
#         media_type = 'video'
#         media_url = data['video']
    
#     try:
#         analysis = analyze_media(media_type, media_url, text_msg)
        
#         return jsonify(analysis), 200
    
#     except Exception as e:
#         logging.error(f"Analysis error: {e}")
#         return jsonify({"error": str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Endpoint for media analysis
    """
    data = request.get_json()
    
    # Enhanced logging for debugging
    logging.info(f"Received request data: {data}")
    
    # Validate request structure
    if 'type' not in data or 'media' not in data:
        logging.error("Invalid request structure")
        return jsonify({"error": "Invalid request structure. Requires 'type' and 'media'."}), 400
    
    media_type = data['type']
    media_url = data['media']
    text_msg = data.get('text_msg', '')
    
    # More robust URL validation
    try:
        result = urlparse(media_url)
        if not all([result.scheme, result.netloc]):
            logging.error(f"Invalid URL: {media_url}")
            return jsonify({"error": "Invalid media URL"}), 400
    except Exception as e:
        logging.error(f"URL parsing error: {e}")
        return jsonify({"error": "Invalid media URL"}), 400
    
    # Validate media type
    if media_type not in ['video', 'image']:
        logging.error(f"Invalid media type: {media_type}")
        return jsonify({"error": "Media type must be 'video' or 'image'."}), 400
    
    try:
        analysis = analyze_media(media_type, media_url, text_msg)
        
        return jsonify(analysis), 200
    
    except Exception as e:
        logging.error(f"Analysis error: {e}")
        return jsonify({"error": str(e)}), 500
    


if __name__ == '__main__':
    app.run(debug=True, port=3400)