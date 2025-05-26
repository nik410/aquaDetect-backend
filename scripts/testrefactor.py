# import testrefactor as fdp
#
# # Initialize models (do this once)
# fdp.initialize_models()
#
# # Now you can predict
# image_file = "/Users/knewatia/Desktop/p/Pycharm/aquaDetect-backend/testimage.jpeg"
# prediction, confidence, _ = fdp.predict_disease(image_file) # Ignore display_image if not needed
#
# if prediction:
#     print(f"The predicted disease for {image_file} is: {prediction} with confidence: {confidence:.2f}")
# else:
#     print(f"Could not predict disease for {image_file}.")
#
# # If you want to make another prediction:
# # image_file_2 = "path/to/another/image.png"
# # prediction_2, confidence_2, _ = fdp.predict_disease(image_file_2)
# # ...
#

import app as fdp
import matplotlib.pyplot as plt # <-- Import matplotlib

# Initialize models (do this once)
fdp.initialize_models()

# Now you can predict
image_file = "/testimage.jpeg"
prediction, confidence, display_image = fdp.predict_disease(image_file) # Capture display_image

if prediction:
    print(f"The predicted disease for {image_file} is: {prediction} with confidence: {confidence:.2f}")
    if display_image is not None:
        plt.imshow(display_image.astype('uint8'))
        plt.title(f"Prediction: {prediction} ({confidence:.2f})")
        plt.axis('off')
        plt.show() # <-- Add this to display the image
    else:
        print("Could not retrieve image for display.")
else:
    print(f"Could not predict disease for {image_file}.")