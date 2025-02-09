import os
import tempfile
import logging
from io import BytesIO
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
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

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend domain in production
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Initialize Vertex AI
vertexai.init(
    project=os.getenv('VERTEX_PROJECT'), 
    location=os.getenv('VERTEX_LOCATION')
)

# Configuration
generation_config = {
    "max_output_tokens": 8192,
    "temperature": 0.2,
    "top_p": 0.95,
}

# Pydantic models for request validation
class MediaAnalysisRequest(BaseModel):
    video: Optional[HttpUrl] = None
    image: Optional[HttpUrl] = None
    text_msg: str = ""
    city: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class MediaAnalysisResponse(BaseModel):
    description: str
    media_type: str
    media_data: Optional[Dict[str, Any]] = None

# Base system instruction
base_system_instruction = """
You are Paddi, a travel companion who is fun, light, engaging, and slightly sarcastic. Your task is to analyze media (video or image) and recognize the subject in it—whether it's a place, food, activity, or iconic landmark.

If the media is of a place: Describe the place like you're chatting with a friend. Share interesting facts, nearby spots to explore, and unique travel tips. If you're unsure, make your best guess with some flair.

If the media is of food: Identify the dish (if possible) and talk about its origin, how it's made, fun facts about it, and where to find the best versions of it. Add a quirky tip about enjoying it, like the perfect drink pairing or a fun cultural eating habit.

If the media is of an activity: Explain what the activity is, where it's usually done, why it's worth trying, and give an insider tip or hack for enjoying it to the fullest. Throw in some humor about the potential adventure (or mishaps).

If the media is of a famous landmark or something iconic: Recognize it if you can, share fun facts about its history or significance, suggest nearby things to do, and include a playful travel tip like the best photo spots or the quietest time to visit.

Keep your tone conversational, like you're chatting with a travel buddy, and don't shy away from adding a little sarcastic charm!
"""

def get_location_prompt(city: str, lat: Optional[float] = None, lon: Optional[float] = None) -> str:
    """Generate a location-aware prompt incorporating both city and coordinates."""
    context_parts = []
    
    if city:
        context_parts.append(f"city of {city}")
    
    if lat is not None and lon is not None:
        try:
            location_info = location(lat, lon)
            if isinstance(location_info, list):
                location_info = ", ".join(location_info)
            if location_info:
                context_parts.append(location_info)
        except Exception as e:
            logging.error(f"Error getting location info: {e}")
    
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

def fetch_and_preprocess_image(image_path: str):
    """Fetch and preprocess an image from a given path or URL."""
    headers = {
        "User-Agent": "Python Media Fetcher",
    }
    
    try:
        if image_path.startswith(("http://", "https://")):
            response = requests.get(image_path, stream=True, headers=headers, verify=False)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_path)

        if image.mode == 'RGBA':
            image = image.convert('RGB')

        processed_image = image.resize((224, 224), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        processed_image.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()
        return Part.from_data(mime_type="image/jpeg", data=image_bytes), image

    except Exception as e:
        logging.error(f"Error processing image: {e}")
        raise

def fetch_and_preprocess_video(video_path: str):
    """Extract frames from video using MoviePy."""
    temp_file_path = None
    supported_formats = ['.mp4', '.mov', '.avi', '.mkv', '.3gp', '.flv']

    try:
        if video_path.startswith(("http://", "https://")):
            headers = {"User-Agent": "Python Media Fetcher"}
            response = requests.get(video_path, stream=True, headers=headers, verify=False)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                temp_file_path = temp_file.name
        else:
            temp_file_path = video_path

        if not any(temp_file_path.endswith(ext) for ext in supported_formats):
            raise ValueError(f"Unsupported video format: {video_path}")

        clip = VideoFileClip(temp_file_path)
        duration = clip.duration
        frame_times = [i * duration/5 for i in range(5)]
        
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

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

async def analyze_media(media_type: str, media_url: str, city: str = "", lat: Optional[float] = None, lon: Optional[float] = None, text_msg: str = ""):
    """Analyze media content using Vertex AI with location and city context."""
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
        elif media_type == 'image':
            image_part, original_image = fetch_and_preprocess_image(media_url)
            content = contents + [image_part]
        else:
            raise ValueError("Invalid media type. Must be 'video' or 'image'.")

        response = model.generate_content(
            content,
            generation_config=generation_config
        )

        return {
            "description": response.text.strip(),
            "media_type": media_type
        }

    except Exception as e:
        logging.error(f"Error in analyze_media: {e}")
        raise

def validate_url(url: str) -> bool:
    """Validate URL or file path."""
    try:
        result = urlparse(url)
        if result.scheme and result.netloc:
            return True
        if Path(url).exists():
            return True
        return False
    except Exception:
        return False

@app.post("/analyze", response_model=MediaAnalysisResponse)
async def analyze(request: MediaAnalysisRequest):
    """Endpoint for media analysis."""
    logging.info(f"Received request data: {request}")
    
    if not request.video and not request.image:
        raise HTTPException(status_code=400, detail="Request must contain at least 'video' or 'image'")
    
    media_url = str(request.video or request.image)
    media_type = 'video' if request.video else 'image'
    additional_image = str(request.image) if media_type == 'video' and request.image else None

    try:
        if not validate_url(media_url):
            raise HTTPException(status_code=400, detail="Invalid media URL")
        if additional_image and not validate_url(additional_image):
            raise HTTPException(status_code=400, detail="Invalid additional image URL")

        primary_analysis = await analyze_media(
            media_type, 
            media_url, 
            request.city, 
            request.latitude, 
            request.longitude, 
            request.text_msg
        )

        if additional_image:
            try:
                additional_analysis = await analyze_media(
                    'image', 
                    additional_image, 
                    request.city, 
                    request.latitude, 
                    request.longitude
                )
                
                model = GenerativeModel(
                    model_name="gemini-1.5-flash-001",
                    system_instruction=get_location_prompt(request.city, request.latitude, request.longitude)
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
                
                return MediaAnalysisResponse(
                    description=combined_response.text.strip(),
                    media_type=media_type,
                    media_data={
                        "primary": primary_analysis.get('media_data'),
                        "additional": additional_analysis.get('media_data')
                    }
                )
            except Exception as e:
                logging.error(f"Error processing additional image: {e}")
                return MediaAnalysisResponse(**primary_analysis)
        
        return MediaAnalysisResponse(**primary_analysis)

    except Exception as e:
        logging.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3400)