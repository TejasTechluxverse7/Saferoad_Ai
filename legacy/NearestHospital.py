import requests
from currentLocation import get_current_location

def get_nearest_hospitals(location):
    """Find the nearest hospitals using OLA Places API and return a dictionary of name-phone pairs."""
    try:
        base_url = "https://api.olamaps.io/places/v1/nearbysearch/advanced"
        params = {
            "location": location,
            "types": "hospital",
            "radius": 10000,
            "withCentroid": "false",
            "rankBy": "popular",
            "api_key": "iB2XUpL1hLF8DVh1kgJafvsEt1elZzZi9CyNwtlT"
        }

        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an error if request fails
        data = response.json()

        hospitals = {}

        if "predictions" in data:  # Ensure 'predictions' key exists
            for prediction in data["predictions"]:
                name = prediction.get("description")
                phone_number = prediction.get("formatted_phone_number")
                if phone_number and name:
                    hospitals[name] = phone_number

        return hospitals

    except requests.RequestException as e:
        return f"Error in the get_nearest_hospitals function: {e}"


# # location = "28.5439375,77.3304876"
# hospitals = get_nearest_hospitals()
# print(hospitals)
