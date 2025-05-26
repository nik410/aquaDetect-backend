import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet101 # InceptionV3 is also an option
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess_input
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_v3_preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import base64
import io
from PIL import Image
import numpy as np
# from sklearn.metrics import classification_report # Keep for potential evaluation if needed later
import matplotlib.pyplot as plt
import joblib # To load SVM model and scaler
from scipy import ndimage as ndi
from skimage.feature import graycomatrix, graycoprops # For SVM features
from skimage.filters import gabor_kernel # For SVM features

# --- Configuration & Parameters ---
# CNN settings
CNN_IMG_WIDTH, CNN_IMG_HEIGHT = 224, 224
CNN_TARGET_SIZE = (CNN_IMG_WIDTH, CNN_IMG_HEIGHT)

# SVM settings
SVM_IMG_WIDTH, SVM_IMG_HEIGHT = 256, 256
SVM_TARGET_SIZE = (SVM_IMG_WIDTH, SVM_IMG_HEIGHT)

# Model paths
RESNET_MODEL_PATH = 'models/best_resnet101_fish_disease_model.keras'
GOOGLENET_MODEL_PATH = 'models/best_googlenet_fish_disease_model.keras'
SVM_MODEL_PATH = 'models/fish_disease_svm_model.joblib'
SVM_SCALER_PATH = 'models/feature_scaler.joblib'

# --- Ensemble Weights (Adjust as needed) ---
W1_RESNET = 1.0
W2_SVM = 0.0 # Set to 0 if SVM is not performing well or not to be included
W3_GOOGLENET = 0.0 # Set to 0 if GoogLeNet is not performing well or not to be included

# --- Global Variables for Models and Settings ---
resnet_model = None
googlenet_model = None
svm_model = None
svm_scaler = None
disease_types = []
num_classes = 0
svm_reorder_indices = None

# --- Preprocessing Functions ---
def preprocess_image_for_cnn(image_path, target_size, preprocess_func):
    """Loads, resizes, and prepares an image for a CNN model."""
    try:
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        if img_array.shape[2] == 1: img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4: img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        img_batch = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_func(img_batch)
        return img_preprocessed, img_array # Return original array for display
    except Exception as e:
        print(f"Error processing image {image_path} for CNN: {e}")
        return None, None

# --- SVM Feature Extraction Functions ---
GABOR_FREQUENCIES = (0.1, 0.5)
GABOR_THETAS = 4
GLCM_DISTANCES = [1, 3, 5]
GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GLCM_PROPERTIES = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

def convert_image_to_base64(image_array):
    pil_img = Image.fromarray(image_array.astype('uint8'))
    buff = io.BytesIO()
    pil_img.save(buff, format="PNG")
    base64_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    return base64_str

def preprocess_image_for_svm(image_path, target_size):
    """Loads, resizes, converts to HSI (using HSV as proxy), and filters image for SVM features."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Could not load image {image_path}")
        return None
    img_resized_bgr = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)
    img_resized_rgb = cv2.cvtColor(img_resized_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_resized_rgb, cv2.COLOR_RGB2HSV)
    v_channel = img_hsv[:, :, 2]
    v_filtered = cv2.GaussianBlur(v_channel, (5, 5), 0)
    return v_filtered

def extract_gabor_features(image_gray):
    features = []
    for theta_idx in range(GABOR_THETAS):
        theta = theta_idx / float(GABOR_THETAS) * np.pi
        for frequency in GABOR_FREQUENCIES:
            kernel = gabor_kernel(frequency, theta=theta)
            filtered = ndi.convolve(image_gray, np.real(kernel), mode='wrap')
            features.extend([np.mean(filtered), np.std(filtered)])
    return np.array(features)

def extract_glcm_features(image_gray):
    if image_gray.dtype != np.uint8:
        if image_gray.max() <= 1.0 and image_gray.min() >= 0.0:
            img_uint8 = (image_gray * 255).astype(np.uint8)
        else:
            img_uint8 = cv2.normalize(image_gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    else:
        img_uint8 = image_gray
    glcm = graycomatrix(img_uint8, distances=GLCM_DISTANCES, angles=GLCM_ANGLES,
                        levels=256, symmetric=True, normed=True)
    features = []
    for prop in GLCM_PROPERTIES:
        try:
            prop_values = graycoprops(glcm, prop)
            features.append(np.mean(prop_values))
        except Exception as e:
            print(f"Warning: Could not calculate GLCM property '{prop}'. Error: {e}. Using 0.")
            features.append(0)
    return np.array(features)

def extract_svm_features(v_filtered):
    gabor_features = extract_gabor_features(v_filtered)
    glcm_features = extract_glcm_features(v_filtered)
    combined_features = np.hstack((gabor_features, glcm_features))
    return combined_features

# --- Model Initialization Function ---
def initialize_models():
    """Loads all models, scalers, and class definitions. Called once."""
    global resnet_model, googlenet_model, svm_model, svm_scaler, disease_types, num_classes, svm_reorder_indices

    print("Loading models and settings...")
    try:
        resnet_model = tf.keras.models.load_model(RESNET_MODEL_PATH)
        print(f"Loaded ResNet101 model from {RESNET_MODEL_PATH}")
    except Exception as e:
        print(f"Error loading ResNet101 model: {e}. Check path and model integrity.")
        resnet_model = None # Ensure it's None if loading failed

    try:
        googlenet_model = tf.keras.models.load_model(GOOGLENET_MODEL_PATH, compile=False)
        print(f"Loaded GoogLeNet model from {GOOGLENET_MODEL_PATH}")
    except Exception as e:
        print(f"Error loading GoogLeNet model: {e}. Check path and model integrity.")
        googlenet_model = None

    if W2_SVM > 0: # Only load SVM if its weight is positive
        try:
            svm_model = joblib.load(SVM_MODEL_PATH)
            print(f"Loaded SVM model from {SVM_MODEL_PATH}")
        except Exception as e:
            print(f"Error loading SVM model: {e}. Check path.")
            svm_model = None
        try:
            svm_scaler = joblib.load(SVM_SCALER_PATH)
            print(f"Loaded SVM scaler from {SVM_SCALER_PATH}")
        except Exception as e:
            print(f"Error loading SVM scaler: {e}. Check path.")
            svm_scaler = None
    else:
        print("SVM model and scaler not loaded as W2_SVM is 0.")
        svm_model = None
        svm_scaler = None


    # Define Class Order (CRUCIAL)
    disease_types_local = sorted(['Bacterial diseases - Aeromoniasis', 'Fungal diseases Saprolegniasis',
                               'Healthy Fish', 'Parasitic diseases', 'Redspot',
                               'THE BACTERIAL GILL ROT', 'Tail And Fin Rot'])
    disease_types = disease_types_local # Assign to global
    num_classes = len(disease_types)
    print(f"Using class order: {disease_types}")

    if svm_model: # Only configure SVM class order if SVM model is loaded
        try:
            svm_class_order = list(svm_model.classes_)
            print(f"SVM model class order: {svm_class_order}")
            if svm_class_order != disease_types:
                print("WARNING: SVM class order differs from defined 'disease_types'. Probabilities will be reordered.")
                svm_reorder_indices = [svm_class_order.index(cls) for cls in disease_types]
            else:
                svm_reorder_indices = None
        except AttributeError:
            print("Could not read classes_ from SVM model. Assuming order matches 'disease_types'. Cross-check needed!")
            svm_reorder_indices = None
    print("Models and settings initialized.")

# --- Main Prediction Function ---
def predict_disease(image_path):
    """
    Predicts fish disease from an image using an ensemble of models.

    Args:
        image_path (str): The path to the image file.

    Returns:
        tuple: (predicted_disease_name, confidence_score, display_image_array)
               Returns (None, 0.0, None) if prediction fails.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image path '{image_path}' not found.")
        return None, 0.0, None

    # Ensure models are loaded
    if resnet_model is None and googlenet_model is None and svm_model is None:
        print("Error: No models are loaded. Call initialize_models() first or check loading errors.")
        return None, 0.0, None
    if num_classes == 0: # Check if class list is populated
        print("Error: disease_types not initialized. Call initialize_models().")
        return None, 0.0, None

    # --- 1. Preprocess for each model ---
    display_img_for_output = None # To store one of the images for display

    # ResNet
    probs_resnet = np.zeros(num_classes) # Default to zero probabilities
    if resnet_model and W1_RESNET > 0:
        resnet_input, display_img_resnet = preprocess_image_for_cnn(image_path, CNN_TARGET_SIZE, resnet_preprocess_input)
        if resnet_input is not None:
            try:
                probs_resnet = resnet_model.predict(resnet_input)[0]
                if display_img_for_output is None: display_img_for_output = display_img_resnet
                print(f"ResNet Probs (raw): {np.round(probs_resnet, 3)}")
            except Exception as e:
                print(f"Error predicting with ResNet101: {e}")
        else:
            print("ResNet preprocessing failed.")
    else:
        if not resnet_model: print("ResNet model not loaded.")
        if W1_RESNET == 0: print("ResNet weight is 0, skipping prediction.")


    # GoogLeNet
    probs_googlenet = np.zeros(num_classes)
    if googlenet_model and W3_GOOGLENET > 0:
        googlenet_input, display_img_googlenet = preprocess_image_for_cnn(image_path, CNN_TARGET_SIZE, inception_v3_preprocess_input)
        if googlenet_input is not None:
            try:
                probs_googlenet = googlenet_model.predict(googlenet_input)[0]
                if display_img_for_output is None: display_img_for_output = display_img_googlenet
                print(f"GoogLeNet Probs (raw): {np.round(probs_googlenet, 3)}")
            except Exception as e:
                print(f"Error predicting with GoogLeNet: {e}")
        else:
            print("GoogLeNet preprocessing failed.")
    else:
        if not googlenet_model: print("GoogLeNet model not loaded.")
        if W3_GOOGLENET == 0: print("GoogLeNet weight is 0, skipping prediction.")

    # SVM
    probs_svm = np.zeros(num_classes)
    if svm_model and svm_scaler and W2_SVM > 0:
        svm_v_channel = preprocess_image_for_svm(image_path, SVM_TARGET_SIZE)
        if svm_v_channel is not None:
            try:
                svm_features = extract_svm_features(svm_v_channel)
                svm_features_scaled = svm_scaler.transform(svm_features.reshape(1, -1))
                raw_probs_svm = svm_model.predict_proba(svm_features_scaled)[0]
                print(f"SVM Probs (raw): {np.round(raw_probs_svm, 3)}")
                if svm_reorder_indices is not None:
                    probs_svm = raw_probs_svm[svm_reorder_indices]
                    print(f"SVM Probs (reordered): {np.round(probs_svm, 3)}")
                else:
                    probs_svm = raw_probs_svm
            except Exception as e:
                print(f"Error predicting with SVM: {e}")
        else:
            print("SVM preprocessing failed.")
    else:
        if not svm_model or not svm_scaler: print("SVM model or scaler not loaded.")
        if W2_SVM == 0: print("SVM weight is 0, skipping prediction.")


    # --- Sanity Check: Ensure all probability vectors have the correct length ---
    # (Already handled by initializing with np.zeros(num_classes))
    # However, check if at least one model produced valid probabilities
    active_models = 0
    if W1_RESNET > 0 and resnet_model and np.any(probs_resnet): active_models +=1
    if W2_SVM > 0 and svm_model and np.any(probs_svm): active_models +=1
    if W3_GOOGLENET > 0 and googlenet_model and np.any(probs_googlenet): active_models +=1

    if active_models == 0:
        print("No models contributed to the prediction. Cannot combine probabilities.")
        return "Prediction Error", 0.0, display_img_for_output

    # --- 2. Combine Probabilities (Weighted Average) ---
    print(f"\nCombining probabilities with weights: ResNet={W1_RESNET}, SVM={W2_SVM}, GoogLeNet={W3_GOOGLENET}")
    total_weight = W1_RESNET + W2_SVM + W3_GOOGLENET
    if total_weight <= 0:
        print("Warning: Sum of weights is zero or negative. Predictions might be unreliable.")
        # Default to equal weights if all are zero, or handle as error
        if W1_RESNET == 0 and W2_SVM == 0 and W3_GOOGLENET == 0:
            print("All weights are zero. Cannot make a prediction.")
            return "Configuration Error: All weights zero", 0.0, display_img_for_output
        # If some are negative and sum to <=0, this is a config issue
        # For now, proceed, but this should be flagged.
        # A more robust approach might be to normalize only positive weights.

    combined_probs = (W1_RESNET * probs_resnet +
                      W2_SVM * probs_svm +
                      W3_GOOGLENET * probs_googlenet)

    if total_weight > 0 : # Avoid division by zero if all active model weights were zero
        combined_probs = combined_probs / total_weight
    else: # This case should ideally be caught by the check above.
          # If only some weights are zero but others are positive, total_weight will be > 0.
          # This branch implies all active weights summed to zero or negative.
        print("Warning: Total weight is not positive. Using raw sum of weighted probabilities.")


    print(f"Combined Probs: {np.round(combined_probs, 3)}")

    # --- 3. Final Prediction ---
    if not np.any(combined_probs): # Check if combined_probs is all zeros
        print("Combined probabilities are all zero. Cannot determine prediction.")
        return "Prediction Failed (Zero Probs)", 0.0, display_img_for_output

    final_class_index = np.argmax(combined_probs)
    final_confidence = combined_probs[final_class_index]
    final_prediction = disease_types[final_class_index]

    return final_prediction, final_confidence, display_img_for_output


# --- Main Execution Block (Example Usage) ---
if __name__ == "__main__":
    # --- 1. Initialize models and settings once ---
    initialize_models()

    # --- 2. Get image path for prediction ---
    # new_image_path = r'/Users/knewatia/Desktop/p/Pycharm/Integration/whitefungal.jpeg' # Example
    new_image_path = input("Enter the path to the fish image: ")

    if not new_image_path:
        print("No image path provided. Exiting.")
    else:
        print(f"\nPredicting on new image: {new_image_path}")
        predicted_disease, confidence, display_image = predict_disease(new_image_path)

        if predicted_disease:
            print("\n--- Ensemble Prediction ---")
            print(f"-> Predicted Disease: {predicted_disease}")
            print(f"-> Final Confidence Score: {confidence:.4f}")

            if display_image is not None:
                plt.imshow(display_image.astype('uint8')) # Display non-normalized image
                plt.title(f"Prediction: {predicted_disease} ({confidence:.2f})")
                plt.axis('off')
                plt.show()
            else:
                print("Could not display image (preprocessing might have failed for all displayable types).")
        else:
            print("Disease prediction failed.")