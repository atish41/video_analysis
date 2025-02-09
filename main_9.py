import os
import tempfile
import logging
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS  # Add this import
import requests
from PIL import Image
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv
from places import location
from moviepy.editor import VideoFileClip
# from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Add CORS to the app
CORS(app, resources={
    r"/analyze": {
        "origins": ["*"],  # Replace with your frontend domain in production
        "methods": ["POST"],
        "allow_headers": ["Content-Type"]
    }
})


# Initialize Vertex AI
vertexai.init(
    project=os.getenv('VERTEX_PROJECT'), 
    location=os.getenv('VERTEX_LOCATION')
)

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 0.2, 
    "top_p": 0.95,
}

base_system_instruction = """
You are Paddi, a travel companion who is fun, light, engaging, and slightly sarcastic. Your task is to analyze media (video or image) and recognize the subject in it—whether it's a place, food, activity, or iconic landmark.

If the media is of a place: Describe the place like you're chatting with a friend. Share interesting facts, nearby spots to explore, and unique travel tips. If you're unsure, make your best guess with some flair.

If the media is of food: Identify the dish (if possible) and talk about its origin, how it's made, fun facts about it, and where to find the best versions of it. Add a quirky tip about enjoying it, like the perfect drink pairing or a fun cultural eating habit.

If the media is of an activity: Explain what the activity is, where it's usually done, why it's worth trying, and give an insider tip or hack for enjoying it to the fullest. Throw in some humor about the potential adventure (or mishaps).

If the media is of a famous landmark or something iconic: Recognize it if you can, share fun facts about its history or significance, suggest nearby things to do, and include a playful travel tip like the best photo spots or the quietest time to visit.

Keep your tone conversational, like you're chatting with a travel buddy, and don't shy away from adding a little sarcastic charm!
"""




def get_location_prompt(city, lat=None, lon=None):
    """
    Generate a location-aware prompt incorporating both city and coordinates.
    
    Args:
        city (str): City name
        lat (float, optional): Latitude
        lon (float, optional): Longitude
        
    Returns:
        str: Location-aware system prompt
    """
    context_parts = []
    
    # Add city context if provided
    if city:
        context_parts.append(f"city of {city}")
    
    # Add location info from coordinates if provided
    if lat is not None and lon is not None:
        try:
            location_info = location(lat, lon)
            # Handle the case where location_info is a list
            if isinstance(location_info, list):
                # Join list items with commas if it's a list
                location_info = ", ".join(location_info)
            if location_info:
                context_parts.append(location_info)
        except Exception as e:
            logging.error(f"Error getting location info: {e}")
    
    # If we have any location context, create a location-aware prompt
    if context_parts:
        location_context = " and ".join(context_parts)
        return f"""
You are Paddi, analyzing media from the {location_context}.

Based on this location context and what you see in the media:
1. Describe what you see in a friendly, conversational way
2. Incorporate relevant details about the location, including specific city information and surroundings
3. Share specific tips and recommendations for this area
4. Point out any interesting connections between what's in the media and the location
5. If you see landmarks or activities, relate them to this specific place
6. Include local insights about the city and neighborhood when relevant

Keep your tone fun and engaging, like you're excitedly telling a friend about this spot!
"""
    return base_system_instruction

def fetch_and_preprocess_image(image_path):
    """
    Fetch and preprocess an image from a given path or URL
    
    Args:
        image_path (str): Path or URL of the image
    
    Returns:
        tuple: Processed image part for Vertex AI and original image
    """ 
    headers = {
        "User-Agent": "Python Media Fetcher",
    }
    
    try:
        if image_path.startswith(("http://", "https://")):
            try:
                # Disable SSL verification to handle more URLs
                response = requests.get(image_path, stream=True, headers=headers, verify=False)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            except requests.RequestException as e:
                logging.error(f"Error fetching image {image_path}: {e}")
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
        import traceback
        logging.error(traceback.format_exc())
        raise

# def fetch_and_preprocess_video(video_path):
#     """
#     Fetch and preprocess a video from a given path or URL
    
#     Args:
#         video_path (str): Path or URL of the video
    
#     Returns:
#         tuple: List of video frame parts and first frame
#     """
#     temp_file_path = None
#     supported_formats = ['.mp4', '.mov', '.avi', '.mkv', '.3gp', '.flv']

#     try:
#         # Fetch video from URL
#         if video_path.startswith(("http://", "https://")):
#             headers = {
#                 "User-Agent": "Python Media Fetcher",
#             }
#             response = requests.get(video_path, stream=True, headers=headers, verify=False)
            
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".mkv") as temp_file:
#                 for chunk in response.iter_content(chunk_size=8192):
#                     temp_file.write(chunk)
#                 temp_file_path = temp_file.name
#         else:
#             temp_file_path = video_path

#         # Validate file format
#         if not any(temp_file_path.endswith(ext) for ext in supported_formats):
#             logging.error(f"Unsupported video format: {video_path}")
#             raise ValueError(f"Unsupported video format: {video_path}")

#         # Open video capture
#         cap = cv2.VideoCapture(temp_file_path)
#         if not cap.isOpened():
#             logging.error(f"Failed to open video file: {video_path}")
#             raise ValueError(f"Failed to open video file: {video_path}")

#         frames = []
#         success, frame = cap.read()
        
#         # If video is empty
#         if not success:
#             logging.error(f"No frames could be read from the video: {video_path}")
#             raise ValueError("Video appears to be empty or unreadable")

#         # Get total frame count and calculate interval
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         frame_interval = max(1, total_frames // 5)

#         # Convert first frame to PIL Image for returning
#         first_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#         # Collect up to 5 frames
#         frame_count = 0
#         while success and frame_count < 5:
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             pil_image = Image.fromarray(frame)
#             pil_image = pil_image.resize((224, 224), Image.Resampling.LANCZOS)
#             buffer = BytesIO()
#             pil_image.save(buffer, format="JPEG")
#             frames.append(Part.from_data(mime_type="image/jpeg", data=buffer.getvalue()))
            
#             # Move to next frame
#             cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + frame_interval)
#             success, frame = cap.read()
#             frame_count += 1

#         cap.release()
#         return frames, first_frame

#     except Exception as e:
#         logging.error(f"Error processing video: {e}")
#         import traceback
#         logging.error(traceback.format_exc())
#         raise
#     finally:
#         if temp_file_path and temp_file_path != video_path and os.path.exists(temp_file_path):
#             os.remove(temp_file_path)


def fetch_and_preprocess_video(video_path):
    """
    Extract frames from video using MoviePy instead of OpenCV
    """
    temp_file_path = None
    supported_formats = ['.mp4', '.mov', '.avi', '.mkv', '.3gp', '.flv']

    try:
        # Fetch video from URL if needed
        if video_path.startswith(("http://", "https://")):
            headers = {"User-Agent": "Python Media Fetcher"}
            response = requests.get(video_path, stream=True, headers=headers, verify=False)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                temp_file_path = temp_file.name
        else:
            temp_file_path = video_path

        # Validate file format
        if not any(temp_file_path.endswith(ext) for ext in supported_formats):
            raise ValueError(f"Unsupported video format: {video_path}")

        # Load video and extract frames
        clip = VideoFileClip(temp_file_path)
        duration = clip.duration
        frame_times = [i * duration/5 for i in range(5)]  # Get 5 evenly spaced frames
        
        frames = []
        first_frame = None
        
        for time in frame_times:
            frame = clip.get_frame(time)
            pil_image = Image.fromarray(frame)
            pil_image = pil_image.resize((224, 224), Image.Resampling.LANCZOS)
            
            if first_frame is None:
                first_frame = Image.fromarray(frame)
            
            buffer = BytesIO()
            pil_image.save(buffer, format="JPEG")
            frames.append(Part.from_data(mime_type="image/jpeg", data=buffer.getvalue()))

        clip.close()
        return frames, first_frame

    except Exception as e:
        logging.error(f"Error processing video: {e}")
        raise
    finally:
        if temp_file_path and temp_file_path != video_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                logging.error(f"Error removing temporary file: {e}")


def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def analyze_media(media_type, media_url, city="", lat=None, lon=None, text_msg=""):
    """
    Analyze media content using Vertex AI with location and city context
    """
    try:
        system_prompt = get_location_prompt(city, lat, lon)
        model = GenerativeModel(
            model_name="gemini-1.5-flash-001",
            system_instruction=system_prompt
        )

        context_msg = text_msg
        if city:
            context_msg = f"Location: {city}. " + context_msg

        contents = [Part.from_text(f"Please analyze this media. {context_msg}")]

        if media_type == 'video':
            video_frames, first_frame = fetch_and_preprocess_video(media_url)
            content = contents + [video_frames[0]]
            
            response = model.generate_content(
                content,
                generation_config=generation_config
            )
            
            return {
                "description": response.text.strip(),
                "media_type": "video",
                # "media_data": image_to_base64(first_frame)
            }
        
        elif media_type == 'image':
            image_part, original_image = fetch_and_preprocess_image(media_url)
            content = contents + [image_part]
            
            response = model.generate_content(
                content,
                generation_config=generation_config
            )
            
            return {
                "description": response.text.strip(),
                "media_type": "image",
                # "media_data": image_to_base64(original_image)
            }
        else:
            raise ValueError("Invalid media type. Must be 'video' or 'image'.")

    except Exception as e:
        logging.error(f"Error in analyze_media: {e}")
        raise

# ... [Previous configurations and functions remain the same] ...

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Endpoint for media analysis
    """
    # If you prefer to manually set CORS headers instead of using Flask-CORS:
    # response = make_response(jsonify({"message": "Success"}))
    # response.headers.add("Access-Control-Allow-Origin", "*")
    # response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    # response.headers.add("Access-Control-Allow-Methods", "POST")
    
    data = request.get_json()
    
    logging.info(f"Received request data: {data}")
    
    if 'video' not in data and 'image' not in data:
        logging.error("Invalid request structure")
        return jsonify({"error": "Request must contain at least 'video' or 'image'"}), 400
    
    # ... [Rest of the analyze function remains the same] ...
    media_url = data.get('video') or data.get('image')
    media_type = 'video' if 'video' in data else 'image'
    text_msg = data.get('text_msg', '')
    city = data.get('city', '')
    
    # Get latitude and longitude from request
    lat = data.get('latitude')
    lon = data.get('longitude')

    # Convert string coordinates to float if they exist
    if lat is not None and lon is not None:
        try:
            lat = float(lat)
            lon = float(lon)
        except ValueError:
            logging.error("Invalid coordinates format")
            return jsonify({"error": "Invalid coordinates format"}), 400
    
    additional_image = data.get('image') if media_type == 'video' else None

    try:
        # Validate URLs
        if not validate_url(media_url):
            return jsonify({"error": "Invalid media URL"}), 400
        if additional_image and not validate_url(additional_image):
            return jsonify({"error": "Invalid additional image URL"}), 400

        # Process primary media
        primary_analysis = analyze_media(media_type, media_url, city, lat, lon, text_msg)

        # Process additional image if present
        if additional_image:
            try:
                additional_analysis = analyze_media('image', additional_image, city, lat, lon)
                
                # Combine descriptions using the same location context
                model = GenerativeModel(
                    model_name="gemini-1.5-flash-001",
                    system_instruction=get_location_prompt(city, lat, lon)
                )
                
                combined_contents = [
                    Part.from_text("Combine these observations into one engaging story, incorporating the location context:"),
                    Part.from_text(f"Main content: {primary_analysis['description']}"),
                    Part.from_text(f"Additional content: {additional_analysis['description']}")
                ]
                
                combined_response = model.generate_content(
                    combined_contents,
                    generation_config=generation_config
                )
                
                response = {
                    "description": combined_response.text.strip(),
                    "media_type": media_type,
                    "media_data": {
                        "primary": primary_analysis['media_data'],
                        "additional": additional_analysis['media_data']
                    }
                }
            except Exception as e:
                logging.error(f"Error processing additional image: {e}")
                response = primary_analysis
        else:
            response = primary_analysis

        return jsonify(response), 200

    except Exception as e:
        logging.error(f"Analysis error: {e}")
        return jsonify({"error": str(e)}), 500

# @app.route('/analyze', methods=['OPTIONS'])
# def handle_options():
#     """
#     Handle OPTIONS requests for CORS preflight
#     """
#     response = jsonify({"message": "OK"})
#     response.headers.add("Access-Control-Allow-Origin", "*")  # Replace with your frontend domain in production
#     response.headers.add("Access-Control-Allow-Headers", "Content-Type")
#     response.headers.add("Access-Control-Allow-Methods", "POST")
#     return response, 200




def validate_url(url):
    """
    Validate URL or file path
    """
    try:
        result = urlparse(url)
        if result.scheme and result.netloc:
            return True
        if Path(url).exists():
            return True
        return False
    except Exception:
        return False

if __name__ == '__main__':
    app.run(debug=True, port=3400)