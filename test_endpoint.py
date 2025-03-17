import requests
import os

# Define API endpoints
BASE_URL = "http://127.0.0.1:8000"
TOKEN_ENDPOINT = f"{BASE_URL}/count_tokens"
SCORE_ENDPOINT = f"{BASE_URL}/scores"

def test_token_count(text):
    response = requests.post(TOKEN_ENDPOINT, data={"texts": text})
    if response.status_code == 200:
        print("Token Count Response:", response.json())
    else:
        print("Error in token count request:", response.text)

def test_image_score(texts, image_paths):
# Prepare the files
    files = []
    for i, img_path in enumerate(image_paths):
        files.append(
            ('files', (os.path.basename(img_path), open(img_path, 'rb'), 'image/jpeg'))
        )
    
    # Make the API request
    response = requests.post(
        SCORE_ENDPOINT,
        files=files,
        data={"texts": texts}
    )
    
    # Check if request was successful
    if response.status_code == 200:
        print(response.json())
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

# Example usage
if __name__ == "__main__":
    sample_text = ["A photo of a bear.", "A picture of a brown bear with green grass in the background."]
    sample_image_path = ["test_image.png", "test_image.png", "test_image.png", "test_image.png", "test_image.png"]
    
    print("Testing Token Count Endpoint...")
    test_token_count(sample_text)
    
    print("\nTesting Image Score Endpoint...")
    test_image_score(sample_text, sample_image_path)
