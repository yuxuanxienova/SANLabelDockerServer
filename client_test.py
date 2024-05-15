import requests

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
        # Save the returned image from the server
        with open("processed_image.png", 'wb') as output_file:
            output_file.write(response.content)
        print("Image processed successfully")
    else:
        print("Error:", response.text)
