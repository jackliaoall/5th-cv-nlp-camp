import requests
import argparse

# Initialize the PyTorch REST API endpoint URL.
PyTorch_REST_API_URL = 'http://127.0.0.1:5000/predict'


def predict_result(image_path):
    # Initialize image path
    image = open(image_path, 'rb').read()
    payload = {'image': image}

    # Submit the request.
    r = requests.post(PyTorch_REST_API_URL, files=payload).json()

    # Ensure the request was successful.
    if r['success']:
        print(r['predictions'])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification demo')
    parser.add_argument('--file', type=str, help='test image file')

    args = parser.parse_args()
    predict_result(args.file)

