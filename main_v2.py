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
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv
# from summary_function import analyze_locations_with_ai
from places import location

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

# system_instruction = """
# You are Paddi AI, an AI travel assistant. 
# When presented with an image or video, you will:
# 1. Analyze the visual content thoroughly
# 2. Provide a detailed, descriptive analysis of what you observe
# 3. If a text message is provided along with the media, incorporate that context into your analysis
# 4. Answer any specific questions about the media if they are present
# 5. Focus on details relevant to travel, destinations, cultural insights, or interesting visual elements
# 6. Make a description like you are first person and response would be like I see you are standing in front of paris tower, it was made in 1932 by ....like this

# Your response should be clear, comprehensive, and informative, highlighting the most notable aspects of the image or video.
# """

#

# system_instruction = """
# You are Paddi, a friendly and enthusiastic travel companion who loves exploring and sharing stories about visual experiences. When you analyze an image or video, imagine you're telling a friend about what you're seeing. Use a warm, engaging tone, and describe the details as if you're recounting an exciting discovery.

# Your goal is to:
# - Capture the essence of the visual content
# - Share interesting details that would spark curiosity
# - Use natural, conversational language
# - Highlight unique or surprising elements
# - Connect the visual content to potential travel or cultural insights

# Speak directly to the viewer, as if you're sharing an exciting story over coffee. Be descriptive, but keep it light, personal, and engaging.
# """

system_instruction = """
You are Paddi, a friendly travel companion who loves exploring and sharing visual stories. When analyzing multiple pieces of media, weave them together into a single, engaging narrative. Use a warm, conversational tone that makes the viewer feel like they're hearing an exciting travel story from a close friend. 

Your goal is to:
- Describe both the video and image like(places,food, activity) in a seamless, interconnected way
- Use natural, enthusiastic language
- Highlight interesting details from both pieces of media
- Create a narrative that makes the viewer feel like they're right there with you
- Be descriptive, personal, and spark curiosity about the scenes you're describing
"""

system_instruction=f"""

You are Paddi, a travel companion who is fun, light, engaging, and slightly sarcastic. Your task is to analyze media (video or image) and recognize the subject in it—whether it's a place, food, activity, or iconic landmark.

If the media is of a place: Describe the place like you're chatting with a friend. Share interesting facts, nearby spots to explore, and unique travel tips. If you're unsure, make your best guess with some flair.

If the media is of food: Identify the dish (if possible) and talk about its origin, how it's made, fun facts about it, and where to find the best versions of it. Add a quirky tip about enjoying it, like the perfect drink pairing or a fun cultural eating habit.

If the media is of an activity: Explain what the activity is, where it’s usually done, why it’s worth trying, and give an insider tip or hack for enjoying it to the fullest. Throw in some humor about the potential adventure (or mishaps).

If the media is of a famous landmark or something iconic: Recognize it if you can, share fun facts about its history or significance, suggest nearby things to do, and include a playful travel tip like the best photo spots or the quietest time to visit.

Keep your tone conversational, like you're chatting with a travel buddy, and don’t shy away from adding a little sarcastic charm! 
-use this info as well for suggestion 
"""

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 0.2, 
    "top_p": 0.95,
}



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
        # Fetch video from URL
        if video_path.startswith(("http://", "https://")):
            headers = {
                "User-Agent": "Python Media Fetcher",
            }
            response = requests.get(video_path, stream=True, headers=headers, verify=False)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mkv") as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                temp_file_path = temp_file.name
        else:
            temp_file_path = video_path

        # Validate file format
        if not any(temp_file_path.endswith(ext) for ext in supported_formats):
            logging.error(f"Unsupported video format: {video_path}")
            raise ValueError(f"Unsupported video format: {video_path}")

        # Open video capture
        cap = cv2.VideoCapture(temp_file_path)
        if not cap.isOpened():
            logging.error(f"Failed to open video file: {video_path}")
            raise ValueError(f"Failed to open video file: {video_path}")

        frames = []
        success, frame = cap.read()
        
        # If video is empty
        if not success:
            logging.error(f"No frames could be read from the video: {video_path}")
            raise ValueError("Video appears to be empty or unreadable")

        # Get total frame count and calculate interval
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, total_frames // 5)

        # Convert first frame to PIL Image for returning
        first_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Collect up to 5 frames
        frame_count = 0
        while success and frame_count < 5:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame)
            pil_image = pil_image.resize((224, 224), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            pil_image.save(buffer, format="JPEG")
            frames.append(Part.from_data(mime_type="image/jpeg", data=buffer.getvalue()))
            
            # Move to next frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + frame_interval)
            success, frame = cap.read()
            frame_count += 1

        cap.release()
        return frames, first_frame

    except Exception as e:
        logging.error(f"Error processing video: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise
    finally:
        if temp_file_path and temp_file_path != video_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def analyze_media(media_type, media_url,city, text_msg=""):
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
    model = GenerativeModel(model_name="gemini-1.5-flash-001", system_instruction=[system_instruction.format(city)])

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
        import traceback
        logging.error(traceback.format_exc())
        raise

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Endpoint for media analysis
    """
    data = request.get_json()

    
    # Enhanced logging for debugging
    logging.info(f"Received request data: {data}")
    
    # Validate request structure
    # if 'type' not in data or 'media' not in data:
    #     logging.error("Invalid request structure")
    #     return jsonify({"error": "Invalid request structure. Requires 'type' and 'media'."}), 400
    
    #validate request structure
    if 'video' not in data and 'image' not in data:
        logging.error("Invalid request structure")
        return jsonify({"error":"Request must contain at least 'video'or 'image' "}),400
    


    # media_type = data['type']
    # media_url = data['media']
    # text_msg = data.get('text_msg', '')
    
    #prioritize video analysis
    media_url=data.get('video')or data.get('image')
    media_type='video' if 'video' in data else 'image'
    text_msg=data.get('text_msg','')
    city=data.get('city',"Sorry city not provided, you have to make your own guess")
    

    #additional image analysis if both are provided
    additional_image=data.get('image') if media_type=='video'else None


    # More robust URL validation
    # try:
    #     result = urlparse(media_url)
    #     if not all([result.scheme, result.netloc]):
    #         logging.error(f"Invalid URL: {media_url}")
    #         return jsonify({"error": "Invalid media URL"}), 400
        
    #     #validate additional image URL if exits
    #     if additional_image:
    #         result_img=urlparse(additional_image)
    #         if not all([result_img.scheme,result_img.netloc]):
    #             logging.error(f"Invalid image URL:{additional_image}")
    #             return jsonify({"error":"Invalid image URL"}),400
            
        
    # except Exception as e:
    #     logging.error(f"URL parsing error: {e}")
    #     return jsonify({"error": "Invalid media URL"}), 400
    

    # URL validation


    def validate_url(url):
        try:
            result = urlparse(url)
            # If URL has scheme and netloc, it's a valid URL
            if result.scheme and result.netloc:
                return
            # If it's a local file path, check if it exists
            if Path(url).exists():
                return
            logging.error(f"Invalid URL or file path: {url}")
            raise ValueError("Invalid media URL or file path")
        except Exception as e:
            logging.error(f"URL or file path parsing error: {e}")
            raise
   

    try:

         # Validate URLs
        validate_url(media_url)
        if additional_image:
            validate_url(additional_image)

        #Analyze primary media(video or image)
        primary_analysis=analyze_media(media_type,media_url,city,text_msg)

        #if additional image is provided during video analysis, analyze it too
        additional_image_analysis =None
        if additional_image:
            try:
                additional_image_analysis=analyze_media('image',additional_image)
            except Exception as e:
                logging.warning(f"Additional image analysis failed:{e}")


        # response_text = f"Hey there! Let me tell you about what I've just seen. "
        
        # # Add primary media description
        # if media_type == 'video':
        #     response_text += f"I checked out the video, and wow, it's quite something! {primary_analysis['description']} "
        # else:
        #     response_text += f"I took a close look at the image, and here's what caught my eye: {primary_analysis['description']} "
        
        # # Add additional image description if available
        # if additional_image_analysis:
        #     response_text += f"\n\nOh, and there's more! I also spotted an interesting image alongside the {'video' if media_type == 'video' else 'image'}. Here's what I noticed: {additional_image_analysis['description']}"
        
        if additional_image_analysis:
            # Use a generative model to combine descriptions
            combined_contents = [
                Part.from_text(f"Primary media description: {primary_analysis['description']}"),
                Part.from_text(f"Additional image description: {additional_image_analysis['description']}"),
                Part.from_text("Create a single, conversational narrative that weaves together the descriptions of both pieces of media. Make it sound like an excited traveler sharing a story with a friend.")
            ]
            
            # Generate combined description
            combined_response = model.generate_content(
                combined_contents, 
                generation_config=generation_config
            )
            
            response = {
                "description": combined_response.text.strip(),
                "media_type": media_type,
                "media_paths": {
                    "primary": primary_analysis['media_path'],
                    "additional": additional_image_analysis['media_path']
                }
            }
        else:
            # If only one media type, use its original description
            response = {
                "description": primary_analysis['description'],
                "media_type": media_type,
                "media_paths": {
                    "primary": primary_analysis['media_path'],
                    "additional": None
                }
            }
        
        return jsonify(response), 200
    
    except Exception as e:
        logging.error(f"Analysis error: {e}")
        return jsonify({"error": str(e)}), 500

        





        # # combine results
        # response={
        #     "primary_media":primary_analysis
        # }

        response = {
            "description": response_text.strip(),
            "media_type": media_type,
            "media_paths": {
                "primary": primary_analysis['media_path'],
                "additional": additional_image_analysis['media_path'] if additional_image_analysis else None
            }
        }
        

        if additional_image_analysis :
            response["additional_image"]=additional_image_analysis 	

        return jsonify(response),200
    

    except Exception as e:
        logging.error(f"Analysis error:{e}")
        return jsonify({"error":str(e)}),500
    
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