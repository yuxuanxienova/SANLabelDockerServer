import requests
from requests_toolbelt.multipart import decoder
import json
from PIL import Image
import numpy as np
import io
import base64


class sanSetOfMaskClient():
    def __init__(self) -> None:
        self.server_url = "http://localhost:5000/image"
        pass
    def sanProcessImage(self,image:Image):
        # Save the PIL image to a byte buffer
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")  # You can change the format to PNG if preferred
        buffer.seek(0)

        files = {'image_buffer': buffer}
        
        # Send a POST request to the Flask server with the image file
        response = requests.post(self.server_url, files=files)
        
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Decode the multipart response
            multipart_data = decoder.MultipartDecoder.from_response(response)
            
            # Iterate through the parts to find the image and the JSON data
            for part in multipart_data.parts:
                content_disposition = part.headers[b'Content-Disposition'].decode()
                if 'filename="processed_image.jpg"' in content_disposition:
                    # Convert the returned image bytes to a PIL Image object
                    processed_image = Image.open(io.BytesIO(part.content))
                    
                    # Save the returned image from the server
                    with open("./Result/processed_image.jpg", 'wb') as output_file:
                        output_file.write(part.content)
                    print("Image processed successfully")
                elif 'filename="mask_data.json"' in content_disposition:
                    # Parse the JSON data
                    mask_data = part.content.decode()
                    maskNumberToPixelCorDict = json.loads(mask_data)
                    print("Mask data received:", maskNumberToPixelCorDict)

            return processed_image, maskNumberToPixelCorDict
        else:
            print("Error:", response.text)
            return None, None
        
if __name__ == "__main__":
    image_path = "./example_data/fp2.jpg"

    image = Image.open(image_path)

    client = sanSetOfMaskClient()
    processed_image, maskNumberToPixelCorDict = client.sanProcessImage(image)