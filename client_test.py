import requests
from requests_toolbelt.multipart import decoder
import json
# URL of the Flask server
server_url = "http://localhost:5000"

# Path to the image file you want to send
image_file = "fp2.png"

# Open the image file in binary mode and read its content
with open(image_file, 'rb') as f:
    files = {'file': f}
    
    # Send a POST request to the Flask server with the image file
    response = requests.post(server_url, files=files)
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Decode the multipart response
        multipart_data = decoder.MultipartDecoder.from_response(response)
        
        # Iterate through the parts to find the image and the JSON data
        for part in multipart_data.parts:
            content_disposition = part.headers[b'Content-Disposition'].decode()
            if 'filename="processed_image.jpg"' in content_disposition:
                # Save the returned image from the server
                with open("processed_image.jpg", 'wb') as output_file:
                    output_file.write(part.content)
                print("Image processed successfully")
            elif 'filename="mask_data.json"' in content_disposition:
                # Parse the JSON data
                mask_data = part.content.decode()
                mask_dict = json.loads(mask_data)
                print("Mask data received:", mask_dict)
    else:
        print("Error:", response.text)
