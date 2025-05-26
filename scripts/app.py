import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet101, InceptionV3
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess_input
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_v3_preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from sklearn.metrics import classification_report # Keep for potential evaluation if needed later
import matplotlib.pyplot as plt
import joblib # To load SVM model and scaler
from scipy import ndimage as ndi
from skimage.feature import graycomatrix, graycoprops # For SVM features
from skimage.filters import gabor_kernel # For SVM features

# --- Configuration & Parameters ---
# CNN settings (use the settings the models were trained with)
CNN_IMG_WIDTH, CNN_IMG_HEIGHT = 224, 224
CNN_TARGET_SIZE = (CNN_IMG_WIDTH, CNN_IMG_HEIGHT)

# SVM settings (use the settings the SVM model was trained with)
SVM_IMG_WIDTH, SVM_IMG_HEIGHT = 256, 256 # From original SVM code example
SVM_TARGET_SIZE = (SVM_IMG_WIDTH, SVM_IMG_HEIGHT)

# Model paths (Update these to your actual saved model locations)
RESNET_MODEL_PATH = '../models/best_resnet101_fish_disease_model.keras'  # Or resnet101_fish_disease_model.keras
GOOGLENET_MODEL_PATH = '../models/best_googlenet_fish_disease_model.keras'  # Or googlenet_fish_disease_model.keras
SVM_MODEL_PATH = '../models/fish_disease_svm_model.joblib'  #
SVM_SCALER_PATH = '../models/feature_scaler.joblib'  #

# --- Ensemble Weights (Adjust based on model performance or tuning) ---
# Example: Equal weights
W1_RESNET = 1.0 #
W2_SVM = 0.0 #
W3_GOOGLENET = 0.0 #
# Example: Weights based on hypothetical validation accuracy
# W1_RESNET = 0.95
# W2_SVM = 0.85 # SVM on handcrafted features often lower
# W3_GOOGLENET = 0.92

# --- Preprocessing Functions ---
def preprocess_image_for_cnn(image_path, target_size, preprocess_func): #
    """Loads, resizes, and prepares an image for a CNN model.""" #
    try:
        # print(f"Preprocessing CNN: {image_path}") # Optional print
        img = load_img(image_path, target_size=target_size) #
        img_array = img_to_array(img) #
        if img_array.shape[2] == 1: img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB) #
        elif img_array.shape[2] == 4: img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB) #
        img_batch = np.expand_dims(img_array, axis=0) #
        img_preprocessed = preprocess_func(img_batch) # Use the specific preprocess_input
        return img_preprocessed, img_array #
    except Exception as e:
        print(f"Error processing image {image_path} for CNN: {e}") #
        return None, None #

# --- SVM Feature Extraction Functions (Copied/Adapted from original SVM code) ---
# Gabor filter parameters (should match parameters used for training SVM)
GABOR_FREQUENCIES = (0.1, 0.5) #
GABOR_THETAS = 4 #

# GLCM parameters (should match parameters used for training SVM)
GLCM_DISTANCES = [1, 3, 5] #
GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4] #
GLCM_PROPERTIES = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'] #

def preprocess_image_for_svm(image_path, target_size): #
    """Loads, resizes, converts to HSI (using HSV as proxy), and filters image for SVM features.""" #
    # print(f"Preprocessing SVM: {image_path}") # Optional print
    img_bgr = cv2.imread(image_path) #
    if img_bgr is None:
        print(f"Error: Could not load image {image_path}") #
        return None #
    img_resized_bgr = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA) #
    img_resized_rgb = cv2.cvtColor(img_resized_bgr, cv2.COLOR_BGR2RGB) #
    img_hsv = cv2.cvtColor(img_resized_rgb, cv2.COLOR_RGB2HSV) #
    v_channel = img_hsv[:, :, 2] #
    v_filtered = cv2.GaussianBlur(v_channel, (5, 5), 0) # Use same filtering as during training
    # Return only the filtered Value channel needed for feature extraction
    return v_filtered #

def extract_gabor_features(image_gray): #
    """Extracts Gabor features (matching original code).""" #
    features = [] #
    for theta_idx in range(GABOR_THETAS): #
        theta = theta_idx / float(GABOR_THETAS) * np.pi #
        for frequency in GABOR_FREQUENCIES: #
            kernel = gabor_kernel(frequency, theta=theta) #
            filtered = ndi.convolve(image_gray, np.real(kernel), mode='wrap') #
            features.append(np.mean(filtered)) #
            features.append(np.std(filtered)) #
    return np.array(features) #

def extract_glcm_features(image_gray): #
    """Extracts GLCM features (matching original code).""" #
    if image_gray.dtype != np.uint8: #
        if image_gray.max() <= 1.0 and image_gray.min() >= 0.0: #
            img_uint8 = (image_gray * 255).astype(np.uint8) #
        else:
            img_uint8 = cv2.normalize(image_gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) #
    else:
        img_uint8 = image_gray #
    glcm = graycomatrix(img_uint8, distances=GLCM_DISTANCES, angles=GLCM_ANGLES, #
                        levels=256, symmetric=True, normed=True) #
    features = [] #
    for prop in GLCM_PROPERTIES: #
        try:
            prop_values = graycoprops(glcm, prop) #
            features.append(np.mean(prop_values)) #
        except Exception as e:
            print(f"Warning: Could not calculate GLCM property '{prop}'. Error: {e}. Using 0.") #
            features.append(0) #
    return np.array(features) #

def extract_svm_features(v_filtered): #
    """Combines Gabor and GLCM feature extraction for SVM.""" #
    gabor_features = extract_gabor_features(v_filtered) #
    glcm_features = extract_glcm_features(v_filtered) #
    combined_features = np.hstack((gabor_features, glcm_features)) #
    return combined_features #

# --- Main Ensemble Prediction Workflow ---
if __name__ == "__main__":
    # --- 1. Load All Models and Scaler ---
    print("Loading models...") #
    try:
        resnet_model = tf.keras.models.load_model(RESNET_MODEL_PATH) #
        print(f"Loaded ResNet101 model from {RESNET_MODEL_PATH}") #
    except Exception as e:
        print(f"Error loading ResNet101 model: {e}. Exiting.") #
        exit() #
    try:
        googlenet_model = tf.keras.models.load_model(GOOGLENET_MODEL_PATH,compile=False) #
        print(f"Loaded GoogLeNet model from {GOOGLENET_MODEL_PATH}") #
    except Exception as e:
        print(f"Error loading GoogLeNet model: {e}. Exiting.") #
        exit() #
    try:
        svm_model = joblib.load(SVM_MODEL_PATH) #
        print(f"Loaded SVM model from {SVM_MODEL_PATH}") #
    except Exception as e:
        print(f"Error loading SVM model: {e}. Exiting.") #
        exit() #
    try:
        svm_scaler = joblib.load(SVM_SCALER_PATH) #
        print(f"Loaded SVM scaler from {SVM_SCALER_PATH}") #
    except Exception as e:
        print(f"Error loading SVM scaler: {e}. Exiting.") #
        exit() #

    # --- 2. Define Class Order (CRUCIAL) ---
    # This order MUST match the output order of your models.
    # Keras models usually follow alphabetical directory order.
    # SVM model order is in svm_model.classes_
    # Find a common order. Let's assume alphabetical based on common practice for Keras:
    # You *MUST* verify this based on how your models were trained.
    # Example: If your class directories were 'Argulus', 'Bacterial_gill_rot', 'Healthy', etc.
    disease_types = sorted(['Bacterial diseases - Aeromoniasis', 'Fungal diseases Saprolegniasis', 'Healthy Fish', 'Parasitic diseases', #
                           'Redspot', 'THE BACTERIAL GILL ROT', 'Tail And Fin Rot']) # Make sure this list is correct and ordered
    num_classes = len(disease_types) #
    print(f"Using class order: {disease_types}") #

    # --- Verify SVM class order ---
    try:
        svm_class_order = list(svm_model.classes_) #
        print(f"SVM model class order: {svm_class_order}") #
        if svm_class_order != disease_types: #
            print("WARNING: SVM class order differs from defined 'disease_types'. Probabilities will be reordered.") #
            # We will need to reorder SVM probabilities later if they differ.
            svm_reorder_indices = [svm_class_order.index(cls) for cls in disease_types] #
        else:
            svm_reorder_indices = None #
    except AttributeError:
        print("Could not read classes_ from SVM model. Assuming order matches 'disease_types'. Cross-check needed!") #
        svm_reorder_indices = None #

    # --- 3. Prepare Input Image ---
    new_image_path = r'/Users/knewatia/Desktop/p/Pycharm/Integration/whitefungal.jpeg' # CHANGE THIS
    print(f"\nPredicting on new image: {new_image_path}") #
    if not os.path.exists(new_image_path): #
        print(f"Error: New image path '{new_image_path}' not found. Cannot predict.") #
        exit() #

    # Preprocess for ResNet
    resnet_input, display_img_resnet = preprocess_image_for_cnn(new_image_path, CNN_TARGET_SIZE, resnet_preprocess_input) #
    # Preprocess for GoogLeNet
    googlenet_input, display_img_googlenet = preprocess_image_for_cnn(new_image_path, CNN_TARGET_SIZE, inception_v3_preprocess_input) #
    # Preprocess for SVM
    svm_v_channel = preprocess_image_for_svm(new_image_path, SVM_TARGET_SIZE) #

    # Check if preprocessing was successful
    if resnet_input is None or googlenet_input is None or svm_v_channel is None: #
        print("Preprocessing failed for one or more models. Exiting.") #
        exit() #

    # Extract and Scale Features for SVM
    svm_features = extract_svm_features(svm_v_channel) #
    svm_features_scaled = svm_scaler.transform(svm_features.reshape(1, -1)) # Reshape for single sample

    # --- 4. Get Predictions (Probabilities) ---
    print("Getting predictions from individual models...") #
    try:
        probs_resnet = resnet_model.predict(resnet_input)[0] # Get the probability array for the single image
        print(f"ResNet Probs (raw): {np.round(probs_resnet, 3)}") #
    except Exception as e:
        print(f"Error predicting with ResNet101: {e}") #
        exit() #
    try:
        probs_googlenet = googlenet_model.predict(googlenet_input)[0] #
        print(f"GoogLeNet Probs (raw): {np.round(probs_googlenet, 3)}") #
    except Exception as e:
        print(f"Error predicting with GoogLeNet: {e}") #
        exit() #
    try:
        probs_svm = svm_model.predict_proba(svm_features_scaled)[0] #
        print(f"SVM Probs (raw): {np.round(probs_svm, 3)}") #
        # --- 5. Align SVM Probabilities (if needed) ---
        if svm_reorder_indices is not None: #
            print("Reordering SVM probabilities...") #
            probs_svm = probs_svm[svm_reorder_indices] #
            print(f"SVM Probs (reordered): {np.round(probs_svm, 3)}") #
    except Exception as e:
        print(f"Error predicting with SVM: {e}") #
        exit() #

    # --- Sanity Check: Ensure all probability vectors have the correct length ---
    if not (len(probs_resnet) == num_classes and len(probs_googlenet) == num_classes and len(probs_svm) == num_classes): #
        print(f"Error: Probability vector lengths mismatch! ResNet={len(probs_resnet)}, GoogLeNet={len(probs_googlenet)}, SVM={len(probs_svm)}. Expected {num_classes}.") #
        exit() #

    # --- 6. Combine Probabilities (Weighted Average) ---
    print(f"\nCombining probabilities with weights: ResNet={W1_RESNET}, SVM={W2_SVM}, GoogLeNet={W3_GOOGLENET}") #
    # Ensure weights sum to non-zero
    total_weight = W1_RESNET + W2_SVM + W3_GOOGLENET #
    if total_weight <= 0: #
        print("Error: Sum of weights must be positive. Setting to equal weights.") #
        W1_RESNET, W2_SVM, W3_GOOGLENET = 1.0, 1.0, 1.0 #
        total_weight = 3.0 #
    combined_probs = (W1_RESNET * probs_resnet + #
                      W2_SVM * probs_svm +       #
                      W3_GOOGLENET * probs_googlenet) / total_weight #
    print(f"Combined Probs: {np.round(combined_probs, 3)}") #

    # --- 7. Final Prediction ---
    final_class_index = np.argmax(combined_probs) #
    final_confidence = combined_probs[final_class_index] # Confidence is the combined probability of the chosen class
    final_prediction = disease_types[final_class_index] # Map index to name

    # --- 8. Display Result ---
    print("\n--- Ensemble Prediction ---") #
    print(f"-> Predicted Disease Index: {final_class_index}") #
    print(f"-> Predicted Disease: {final_prediction}") #
    print(f"-> Final Confidence Score: {final_confidence:.4f}") #

    # Display the image (use one of the display images preprocessed for CNNs)
    plt.imshow(display_img_resnet.astype('uint8')) # Display non-normalized image
    plt.title(f"Ensemble Prediction: {final_prediction} ({final_confidence:.2f})") #
    plt.axis('off') #
    plt.show() #