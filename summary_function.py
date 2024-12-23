import os
import logging
from vertexai.generative_models import GenerativeModel, Part
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Initialize Vertex AI
try:
    import vertexai
    vertexai.init(
        project=os.getenv('VERTEX_PROJECT'), 
        location=os.getenv('VERTEX_LOCATION')
    )
except Exception as e:
    logging.error(f"Error initializing Vertex AI: {e}")
    raise

def analyze_locations_with_ai(locations_list):
    """
    Use generative AI to analyze a list of locations and provide a summary
    
    Args:
        locations_list (list): List of location names/descriptions
        
    Returns:
        str: AI-generated analysis and summary of the area
    """
    try:
        # System instruction for the AI
        system_instruction = """
        You are an expert location and area analyzer. Given a list of establishments:
        3. Provide insights about the neighborhood
        4. Generate a comprehensive yet concise summary of the area
        Focus on providing practical insights about the area's amenities and character.
        -
        """

        # Configuration for the AI model
        generation_config = {
            "max_output_tokens": 400,
            "temperature": 0.3,
            "top_p": 0.8,
        }

        # Initialize the model
        model = GenerativeModel(
            model_name="gemini-1.5-flash-001",
            generation_config=generation_config,
            system_instruction=[system_instruction]
        )

        # Prepare the prompt
        locations_text = "\n".join([f"- {loc}" for loc in locations_list])
        prompt = f"""
        Please analyze these establishments in an area and provide a summary:

        {locations_text}

        Please describe:
        1. What type of area this appears to be
        2. What amenities and services are available
        3. What this suggests about the neighborhood
        """

        # Generate the analysis
        response = model.generate_content(
            Part.from_text(prompt)
        )

        return response.text.strip()

    except Exception as e:
        logging.error(f"Error in location analysis: {e}")
        return f"Error analyzing locations: {str(e)}"

# Example usage:
if __name__ == "__main__":
    locations = [
        'Lenskart.com at Govind Nagar, Nashik',
        'Bhamare Hospital',
        'The Nasikroad Deolali Vyapari Sahakari Bank Ltd.',
        'Gananayak Apartment...',
        'Dr. Sarala Patil - Best dermatologist in Nashik | Skin Specialist in nashik | Skin Doctor in Nashik | Skin Treatment'
    ]
    
    summary = analyze_locations_with_ai(locations)
    print(summary)