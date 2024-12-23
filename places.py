import requests
from pprint import pprint
from summary_function import analyze_locations_with_ai
import os
from dotenv import load_dotenv
load_dotenv()


def location(latitude,longitue):
    url = os.getenv("places_api")

    payload = {
        "maxResultCount": 5,
        "locationRestriction": {"circle": {
                "center": {
                    "latitude":latitude,
                    "longitude": longitue
                },
                "radius": 50
            }}
    }
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": "AIzaSyAOUDmp4iNuX0KE_L2kuR0mHcNWWu3wxjE",
        "X-Goog-FieldMask": "places.displayName"
    }

    response = requests.request("POST", url, json=payload, headers=headers)
    data=response.json()
    # pprint(data)

    all_list=[]

    details=data["places"]
    for item in details:
        all_list.append(item['displayName']['text'])


    return (all_list)

if __name__=="__main__":
    result=location(19.9843,73.7787)
    pprint(result)

