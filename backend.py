from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins, or specify your frontend origin for security

# Define a directory to save uploaded images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# This is a placeholder for your actual Python image processing logic
def process_image_with_python_file(image_path):
    #this is where we will include the image to be passed to out model
    return {"status": "success", "message": f"Image '{os.path.basename(image_path)}' processed successfully."}


@app.route('/uploadimg', methods=['POST'])
def upload_image():
    # Check if a file was sent in the request
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']

    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Securely save the file to the UPLOAD_FOLDER
        # You might want to add more robust filename sanitization
        # and unique naming to prevent overwrites or malicious uploads.
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Now, pass the saved image path to your Python processing function
        processing_result = process_image_with_python_file(filepath)

        # You can also delete the temporary file after processing if not needed
        # os.remove(filepath)

        return jsonify({
            "message": f"Image '{filename}' uploaded and processed successfully!",
            "processing_details": processing_result
        }), 200


    return jsonify({"error": "An unexpected error occurred"}), 500


@app.route('/')
def home():
    return "Flask Backend is running!"


if __name__ == '__main__':
    # Run the Flask app
    # host='0.0.0.0' makes the server accessible from other devices on the network
    # debug=True enables debug mode (auto-reloads on code changes, shows detailed errors)
    app.run(debug=True, host='0.0.0.0', port=5000)
