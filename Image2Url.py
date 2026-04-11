import base64


def image_to_data_url(image_path):
    """Convert an image file to a Base64 Data URL."""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    # Get MIME type based on file extension
    mime_type = "image/png" if image_path.endswith(".png") else "image/jpeg"

    return f"data:{mime_type};base64,{encoded_string}"