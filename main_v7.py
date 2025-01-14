import os
import tempfile
import logging
from io import BytesIO
from functools import lru_cache
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from flask import Flask, request, jsonify
import requests
from PIL import Image
import cv2
import numpy as np
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv
from places import location

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
SUPPORTED_VIDEO_FORMATS = {'.mp4', '.mov', '.avi', '.mkv', '.3gp', '.flv'}
IMAGE_SIZE = (224, 224)
MAX_VIDEO_FRAMES = 5
VERTEX_GENERATION_CONFIG = {
    "max_output_tokens": 8192,
    "temperature": 0.2,
    "top_p": 0.95,
}

@dataclass
class LocationContext:
    city: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class MediaProcessor:
    def __init__(self):
        load_dotenv()
        self._init_vertex_ai()
        
    def _init_vertex_ai(self):
        """Initialize Vertex AI with environment variables."""
        vertexai.init(
            project=os.getenv('VERTEX_PROJECT'),
            location=os.getenv('VERTEX_LOCATION')
        )
        
    @staticmethod
    def get_system_prompt(location_ctx: LocationContext) -> str:
        """Generate location-aware system prompt."""
        base_prompt = """
You are Paddi, a travel companion who is fun, light, engaging, and slightly sarcastic. Your task is to analyze media (video or image) and recognize the subject in it—whether it's a place, food, activity, or iconic landmark.

If the media is of a place: Describe the place like you're chatting with a friend. Share interesting facts, nearby spots to explore, and unique travel tips. If you're unsure, make your best guess with some flair.

If the media is of food: Identify the dish (if possible) and talk about its origin, how it's made, fun facts about it, and where to find the best versions of it. Add a quirky tip about enjoying it, like the perfect drink pairing or a fun cultural eating habit.

If the media is of an activity: Explain what the activity is, where it's usually done, why it's worth trying, and give an insider tip or hack for enjoying it to the fullest. Throw in some humor about the potential adventure (or mishaps).

If the media is of a famous landmark or something iconic: Recognize it if you can, share fun facts about its history or significance, suggest nearby things to do, and include a playful travel tip like the best photo spots or the quietest time to visit.

Keep your tone conversational, like you're chatting with a travel buddy, and don't shy away from adding a little sarcastic charm!
"""
        
        if not any([location_ctx.city, location_ctx.latitude, location_ctx.longitude]):
            return base_prompt
            
        context_parts = []
        if location_ctx.city:
            context_parts.append(f"city of {location_ctx.city}")
            
        if location_ctx.latitude is not None and location_ctx.longitude is not None:
            try:
                loc_info = location(location_ctx.latitude, location_ctx.longitude)
                if isinstance(loc_info, list):
                    loc_info = ", ".join(loc_info)
                if loc_info:
                    context_parts.append(loc_info)
            except Exception as e:
                logger.error(f"Error getting location info: {e}")
                
        if not context_parts:
            return base_prompt
            
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
        return base_prompt

    @staticmethod
    def _process_image(image: Image.Image) -> bytes:
        """Process image to standard size and format."""
        processed = image.convert('RGB').resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        buffer = BytesIO()
        processed.save(buffer, format="JPEG")
        return buffer.getvalue()

    @staticmethod
    @lru_cache(maxsize=100)
    def _fetch_url_content(url: str) -> bytes:
        """Fetch content from URL with caching."""
        headers = {"User-Agent": "Python Media Fetcher"}
        response = requests.get(url, stream=True, headers=headers, verify=False)
        response.raise_for_status()
        return response.content

    def process_image(self, image_path: str) -> Tuple[Part, Image.Image]:
        """Process image from path or URL."""
        try:
            if image_path.startswith(("http://", "https://")):
                content = self._fetch_url_content(image_path)
                image = Image.open(BytesIO(content))
            else:
                image = Image.open(image_path)

            image_bytes = self._process_image(image)
            return Part.from_data(mime_type="image/jpeg", data=image_bytes), image

        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            raise

    def process_video(self, video_path: str) -> Tuple[List[Part], Image.Image]:
        """Process video from path or URL."""
        temp_path = None
        try:
            if video_path.startswith(("http://", "https://")):
                content = self._fetch_url_content(video_path)
                temp_path = self._save_temp_file(content, ".mkv")
            else:
                temp_path = video_path

            if not self._is_valid_video_format(temp_path):
                raise ValueError(f"Unsupported video format: {video_path}")

            return self._extract_video_frames(temp_path)

        except Exception as e:
            logger.error(f"Error processing video: {e}", exc_info=True)
            raise
        finally:
            self._cleanup_temp_file(temp_path, video_path)

    def analyze_media(self, media_type: str, media_url: str, location_ctx: LocationContext, text_msg: str = "") -> Dict[str, Any]:
        """Analyze media content using Vertex AI."""
        try:
            system_prompt = self.get_system_prompt(location_ctx)
            model = GenerativeModel(
                model_name="gemini-1.5-flash-001",
                system_instruction=system_prompt
            )

            context_msg = f"Location: {location_ctx.city}. {text_msg}" if location_ctx.city else text_msg
            contents = [Part.from_text(f"Please analyze this media. {context_msg}")]

            if media_type == 'video':
                return self._analyze_video(media_url, model, contents)
            elif media_type == 'image':
                return self._analyze_image(media_url, model, contents)
            else:
                raise ValueError("Invalid media type. Must be 'video' or 'image'.")

        except Exception as e:
            logger.error(f"Error in analyze_media: {e}", exc_info=True)
            raise

    # Helper methods implementation...
    @staticmethod
    def _is_valid_video_format(path: str) -> bool:
        return any(path.endswith(ext) for ext in SUPPORTED_VIDEO_FORMATS)

    @staticmethod
    def _save_temp_file(content: bytes, suffix: str) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(content)
            return temp_file.name

    @staticmethod
    def _cleanup_temp_file(temp_path: Optional[str], original_path: str) -> None:
        if temp_path and temp_path != original_path and os.path.exists(temp_path):
            os.remove(temp_path)

    def _extract_video_frames(self, video_path: str) -> Tuple[List[Part], Image.Image]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        frames = []
        success, frame = cap.read()
        if not success:
            raise ValueError("Video appears to be empty or unreadable")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, total_frames // MAX_VIDEO_FRAMES)
        first_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        frame_count = 0
        while success and frame_count < MAX_VIDEO_FRAMES:
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_bytes = self._process_image(pil_frame)
            frames.append(Part.from_data(mime_type="image/jpeg", data=frame_bytes))
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + frame_interval)
            success, frame = cap.read()
            frame_count += 1

        cap.release()
        return frames, first_frame

    @staticmethod
    def validate_url(url: str) -> bool:
        """
        Validate if the given string is a valid URL or file path.
        
        Args:
            url (str): URL or file path to validate
            
        Returns:
            bool: True if valid URL or existing file path, False otherwise
        """
        try:
            result = urlparse(url)
            if result.scheme and result.netloc:
                return True
            return Path(url).exists()
        except Exception:
            return False
# Flask application setup
app = Flask(__name__)
media_processor = MediaProcessor()

# @app.route('/analyze', methods=['POST'])
# def analyze():
#     """Endpoint for media analysis."""
#     try:
#         data = request.get_json()
#         logger.info(f"Received request data: {data}")

#         if not {'video', 'image'} & set(data.keys()):
#             return jsonify({"error": "Request must contain 'video' or 'image'"}), 400

#         # Extract and validate request data
#         media_url = data.get('video') or data.get('image')
#         if not media_processor.validate_url(media_url):
#             return jsonify({"error": "Invalid media URL"}), 400

#         media_type = 'video' if 'video' in data else 'image'
#         location_ctx = LocationContext(
#             city=data.get('city', ''),
#             latitude=float(data['latitude']) if 'latitude' in data else None,
#             longitude=float(data['longitude']) if 'longitude' in data else None
#         )

#         # Process media and return response
#         response = media_processor.analyze_media(
#             media_type=media_type,
#             media_url=media_url,
#             location_ctx=location_ctx,
#             text_msg=data.get('text_msg', '')
#         )

#         return jsonify(response), 200

#     except Exception as e:
#         logger.error(f"Analysis error: {e}", exc_info=True)
#         return jsonify({"error": str(e)}), 500



@app.route('/analyze', methods=['POST'])
def analyze():
    """Endpoint for media analysis."""
    try:
        data = request.get_json()
        logger.info(f"Received request data: {data}")

        if not {'video', 'image'} & set(data.keys()):
            return jsonify({"error": "Request must contain 'video' or 'image'"}), 400

        # Extract and validate request data
        media_url = data.get('video') or data.get('image')
        # Fix: Use the validate_url method properly
        if not media_processor.validate_url(media_url):
            return jsonify({"error": "Invalid media URL"}), 400

        media_type = 'video' if 'video' in data else 'image'
        location_ctx = LocationContext(
            city=data.get('city', ''),
            latitude=float(data['latitude']) if 'latitude' in data else None,
            longitude=float(data['longitude']) if 'longitude' in data else None
        )

        # Process media and return response
        response = media_processor.analyze_media(
            media_type=media_type,
            media_url=media_url,
            location_ctx=location_ctx,
            text_msg=data.get('text_msg', '')
        )

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=3400)