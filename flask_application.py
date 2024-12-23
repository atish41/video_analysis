import os
import logging
from flask import Flask, jsonify, request
from collections import Counter
import re

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_location_data(locations_list):
    """
    Analyze a list of location names and provide insights about the area
    
    Args:
        locations_list (list): List of location names/descriptions
        
    Returns:
        dict: Analysis results including area type, key establishments, and summary
    """
    try:
        # Initialize analysis containers
        business_types = []
        medical_facilities = []
        financial_institutions = []
        residential_places = []
        
        # Keywords for classification
        medical_keywords = ['hospital', 'doctor', 'clinic', 'dermatologist', 'specialist', 'medical', 'treatment']
        financial_keywords = ['bank', 'sahakari', 'finance', 'atm']
        residential_keywords = ['apartment', 'complex', 'residency', 'housing', 'nagar']
        business_keywords = ['shop', 'store', 'market', 'mall', '.com']
        
        # Analyze each location
        for location in locations_list:
            location_lower = location.lower()
            
            # Classify based on keywords
            if any(keyword in location_lower for keyword in medical_keywords):
                medical_facilities.append(location)
            elif any(keyword in location_lower for keyword in financial_keywords):
                financial_institutions.append(location)
            elif any(keyword in location_lower for keyword in residential_keywords):
                residential_places.append(location)
            elif any(keyword in location_lower for keyword in business_keywords):
                business_types.append(location)
            
        # Determine area characteristics
        area_characteristics = []
        if medical_facilities:
            area_characteristics.append("healthcare hub")
        if financial_institutions:
            area_characteristics.append("financial district")
        if residential_places:
            area_characteristics.append("residential area")
        if business_types:
            area_characteristics.append("commercial zone")
            
        # Create area summary
        area_type = " and ".join(area_characteristics) if area_characteristics else "mixed-use area"
        
        # Generate analysis results
        analysis = {
            "area_type": area_type,
            "key_establishments": {
                "medical_facilities": medical_facilities,
                "financial_institutions": financial_institutions,
                "residential_places": residential_places,
                "businesses": business_types
            },
            "summary": f"This appears to be a {area_type} in Nashik with {len(locations_list)} identified establishments. "
                      f"The area features {len(medical_facilities)} medical facilities, "
                      f"{len(financial_institutions)} financial institutions, "
                      f"{len(residential_places)} residential complexes, and "
                      f"{len(business_types)} commercial establishments."
        }
        
        return analysis
        
    except Exception as e:
        logging.error(f"Error analyzing locations: {e}")
        raise

app = Flask(__name__)

@app.route('/analyze-locations', methods=['POST'])
def analyze_locations():
    """
    Endpoint for analyzing location data
    """
    try:
        data = request.get_json()
        
        if not data or 'locations' not in data:
            return jsonify({"error": "Invalid request. Please provide a list of locations."}), 400
            
        locations = data['locations']
        
        if not isinstance(locations, list):
            return jsonify({"error": "Locations must be provided as a list."}), 400
            
        analysis = analyze_location_data(locations)
        return jsonify(analysis), 200
        
    except Exception as e:
        logging.error(f"Analysis error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=3400)