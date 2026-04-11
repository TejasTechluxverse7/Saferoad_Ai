import requests

def get_current_location():
    """Fetch the current location using IP-based geolocation."""
    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        location = data.get("loc", "")  # Format: "latitude,longitude"
        return location
    except Exception as e:
        print(f"Error fetching location: {e}")
        return None