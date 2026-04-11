import requests
import os
from Image2Url import image_to_data_url
from dotenv import load_dotenv

load_dotenv()

# Move configurations to the top so functions can see them
API_URL = os.getenv("API_URL")
RECIEVER_ID = os.getenv("RECIEVER_ID")
HEADERS = {"Content-Type": "application/json"}

def getToken():
    payload = {
        "email": "cctv@gmail.com",
        "password": "123456"
    }

    # Fixed f-string placement
    response = requests.post(f"{API_URL}/auth/login", json=payload, headers=HEADERS)

    try:
        response_json = response.json()
        # Use .get() to avoid KeyError if 'token' is missing
        return response_json.get("token") 
    except requests.exceptions.JSONDecodeError:
        print("❌ Error: Response is not valid JSON!")
        return None

# Get the token first
TOKEN = getToken()

# Now set up cookies
COOKIES = {"jwt": TOKEN}

def send_message(text, image_path=None):
    # Fixed f-string placement
    url = f"{API_URL}/messages/send/{RECIEVER_ID}"

    image_url = None
    if image_path:
        image_url = image_to_data_url(image_path)

    payload = {
        "text": text,
        "image": image_url
    }

    response = requests.post(url, json=payload, headers=HEADERS, cookies=COOKIES)

    print(f"Status: {response.status_code}")
    return response.json()

# Example call
# send_message("Hello, this is an accident", "./avatar.png")