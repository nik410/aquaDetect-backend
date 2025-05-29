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
from sklearn.metrics import classification_report # <<<< UNCOMMENTED
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
                # Ensure all disease_types are in svm_class_order for safe indexing
                if all(cls in svm_class_order for cls in disease_types):
                    svm_reorder_indices = [svm_class_order.index(cls) for cls in disease_types]
                else:
                    print("ERROR: Not all 'disease_types' are present in 'svm_model.classes_'. Cannot reorder. SVM predictions might be incorrect.")
                    svm_reorder_indices = None # Or handle as a critical error
                    # Potentially disable SVM or raise an exception if this is critical
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
                probs_resnet = resnet_model.predict(resnet_input, verbose=0)[0] # Added verbose=0
                if display_img_for_output is None: display_img_for_output = display_img_resnet
                # print(f"ResNet Probs (raw): {np.round(probs_resnet, 3)}") # Optional: for detailed debugging
            except Exception as e:
                print(f"Error predicting with ResNet101 for {os.path.basename(image_path)}: {e}")
        else:
            print(f"ResNet preprocessing failed for {os.path.basename(image_path)}.")
    else:
        if not resnet_model and W1_RESNET > 0: print("ResNet model not loaded.")
        # if W1_RESNET == 0: print("ResNet weight is 0, skipping prediction.") # Can be too verbose for batch


    # GoogLeNet
    probs_googlenet = np.zeros(num_classes)
    if googlenet_model and W3_GOOGLENET > 0:
        googlenet_input, display_img_googlenet = preprocess_image_for_cnn(image_path, CNN_TARGET_SIZE, inception_v3_preprocess_input)
        if googlenet_input is not None:
            try:
                probs_googlenet = googlenet_model.predict(googlenet_input, verbose=0)[0] # Added verbose=0
                if display_img_for_output is None: display_img_for_output = display_img_googlenet
                # print(f"GoogLeNet Probs (raw): {np.round(probs_googlenet, 3)}")
            except Exception as e:
                print(f"Error predicting with GoogLeNet for {os.path.basename(image_path)}: {e}")
        else:
            print(f"GoogLeNet preprocessing failed for {os.path.basename(image_path)}.")
    else:
        if not googlenet_model and W3_GOOGLENET > 0: print("GoogLeNet model not loaded.")
        # if W3_GOOGLENET == 0: print("GoogLeNet weight is 0, skipping prediction.")

    # SVM
    probs_svm = np.zeros(num_classes)
    if svm_model and svm_scaler and W2_SVM > 0:
        svm_v_channel = preprocess_image_for_svm(image_path, SVM_TARGET_SIZE)
        if svm_v_channel is not None:
            try:
                svm_features = extract_svm_features(svm_v_channel)
                svm_features_scaled = svm_scaler.transform(svm_features.reshape(1, -1))
                raw_probs_svm = svm_model.predict_proba(svm_features_scaled)[0]
                # print(f"SVM Probs (raw): {np.round(raw_probs_svm, 3)}")
                if svm_reorder_indices is not None:
                    # Check if raw_probs_svm has enough elements for reordering
                    if len(raw_probs_svm) == len(svm_model.classes_):
                        probs_svm_reordered_temp = np.zeros(num_classes)
                        valid_reorder = True
                        for i, target_cls in enumerate(disease_types):
                            try:
                                source_idx = svm_model.classes_.tolist().index(target_cls)
                                probs_svm_reordered_temp[i] = raw_probs_svm[source_idx]
                            except ValueError:
                                print(f"Warning: Class '{target_cls}' not found in SVM model classes during reordering. Setting prob to 0.")
                                # probs_svm_reordered_temp[i] will remain 0
                                valid_reorder = False # Or handle more gracefully
                        probs_svm = probs_svm_reordered_temp
                        if not valid_reorder:
                             print(f"SVM prob reordering issue for {os.path.basename(image_path)}. SVM classes: {svm_model.classes_}, Target classes: {disease_types}")
                        # print(f"SVM Probs (reordered): {np.round(probs_svm, 3)}")
                    else:
                        print(f"SVM raw probability length mismatch. Expected {len(svm_model.classes_)}, got {len(raw_probs_svm)}. Skipping SVM reorder for {os.path.basename(image_path)}.")
                        # probs_svm remains zeros
                elif len(raw_probs_svm) == num_classes: # If no reorder needed, ensure direct assignment is safe
                    probs_svm = raw_probs_svm
                else:
                    print(f"SVM probability length mismatch. Expected {num_classes}, got {len(raw_probs_svm)} and no reorder map. Skipping SVM for {os.path.basename(image_path)}.")
                    # probs_svm remains zeros

            except Exception as e:
                print(f"Error predicting with SVM for {os.path.basename(image_path)}: {e}")
        else:
            print(f"SVM preprocessing failed for {os.path.basename(image_path)}.")
    else:
        if (not svm_model or not svm_scaler) and W2_SVM > 0: print("SVM model or scaler not loaded.")
        # if W2_SVM == 0: print("SVM weight is 0, skipping prediction.")


    # --- Sanity Check: Ensure all probability vectors have the correct length ---
    active_models = 0
    if W1_RESNET > 0 and resnet_model and np.any(probs_resnet): active_models +=1
    if W2_SVM > 0 and svm_model and np.any(probs_svm): active_models +=1
    if W3_GOOGLENET > 0 and googlenet_model and np.any(probs_googlenet): active_models +=1

    if active_models == 0:
        # print("No models contributed to the prediction. Cannot combine probabilities.") # Verbose for batch
        return "Prediction Error", 0.0, display_img_for_output

    # --- 2. Combine Probabilities (Weighted Average) ---
    # print(f"\nCombining probabilities with weights: ResNet={W1_RESNET}, SVM={W2_SVM}, GoogLeNet={W3_GOOGLENET}")
    total_weight = W1_RESNET + W2_SVM + W3_GOOGLENET
    if total_weight <= 0:
        # print("Warning: Sum of weights is zero or negative. Predictions might be unreliable.")
        if W1_RESNET == 0 and W2_SVM == 0 and W3_GOOGLENET == 0:
            # print("All weights are zero. Cannot make a prediction.")
            return "Configuration Error: All weights zero", 0.0, display_img_for_output

    combined_probs = (W1_RESNET * probs_resnet +
                      W2_SVM * probs_svm +
                      W3_GOOGLENET * probs_googlenet)

    if total_weight > 0 :
        combined_probs = combined_probs / total_weight
    # else:
        # print("Warning: Total weight is not positive. Using raw sum of weighted probabilities.")


    # print(f"Combined Probs: {np.round(combined_probs, 3)}")

    # --- 3. Final Prediction ---
    if not np.any(combined_probs): # Check if combined_probs is all zeros
        # print("Combined probabilities are all zero. Cannot determine prediction.")
        return "Prediction Failed (Zero Probs)", 0.0, display_img_for_output

    final_class_index = np.argmax(combined_probs)
    final_confidence = combined_probs[final_class_index]

    if final_class_index < len(disease_types):
        final_prediction = disease_types[final_class_index]
    else:
        print(f"Error: final_class_index {final_class_index} is out of bounds for disease_types (len {len(disease_types)})")
        return "Prediction Error (Index)", 0.0, display_img_for_output


    return final_prediction, final_confidence, display_img_for_output


# --- Function to Evaluate on Test Set and Generate Report ---
def evaluate_and_generate_report(test_dataset_path):
    """
    Evaluates the ensemble model on a test dataset and prints a classification report.

    Args:
        test_dataset_path (str): Path to the root directory of the test dataset.
                                 Subdirectories should be class names.
    """
    if not os.path.exists(test_dataset_path):
        print(f"Error: Test dataset path '{test_dataset_path}' not found.")
        return

    if not disease_types:
        print("Error: 'disease_types' list is empty. Initialize models first.")
        return

    y_true = []
    y_pred = []
    image_files = []
    true_labels_for_files = []

    print(f"\nEvaluating on test dataset: {test_dataset_path}")
    print(f"Expected class names (from subdirectories): {disease_types}")

    for class_name in disease_types:
        class_dir = os.path.join(test_dataset_path, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: Directory for class '{class_name}' not found at '{class_dir}'. Skipping.")
            continue

        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            if os.path.isfile(image_path) and image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_files.append(image_path)
                true_labels_for_files.append(class_name)

    if not image_files:
        print("No image files found in the test dataset. Check paths and file extensions.")
        return

    print(f"Found {len(image_files)} images to evaluate.")

    for i, image_path in enumerate(image_files):
        true_label = true_labels_for_files[i]
        # print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)} (True: {true_label})") # Verbose
        predicted_disease, confidence, _ = predict_disease(image_path)

        if predicted_disease and predicted_disease in disease_types:
            y_true.append(true_label)
            y_pred.append(predicted_disease)
        elif predicted_disease: # Predicted disease is not in known disease_types (e.g. error string)
            print(f"Warning: Prediction for {os.path.basename(image_path)} was '{predicted_disease}'. Cannot use for report. True label: {true_label}")
            # Optionally, assign a placeholder or skip, depending on how you want to handle errors
        else: # Prediction failed (None)
            print(f"Warning: Prediction failed for {os.path.basename(image_path)}. True label: {true_label}")


        if (i + 1) % 50 == 0: # Progress update
            print(f"Processed {i+1}/{len(image_files)} images...")


    if not y_true or not y_pred:
        print("No valid predictions were made. Cannot generate classification report.")
        return

    print("\n--- Ensemble Classification Report ---")
    # Ensure target_names are correctly ordered as per disease_types
    # and that all unique labels in y_true and y_pred are covered.
    unique_labels = sorted(list(set(y_true + y_pred)))

    # Filter disease_types to only include labels present in the results, maintaining original order
    report_target_names = [dt for dt in disease_types if dt in unique_labels]

    # If there are labels in y_true/y_pred not in disease_types (e.g. error strings),
    # classification_report might handle them or error. It's best if y_pred only contains valid class labels.
    # For robustness, ensure all labels in y_true and y_pred are in report_target_names if possible.
    # However, classification_report can infer labels if target_names is not exhaustive.
    # For clarity, using the globally defined disease_types is generally good if all predictions are valid classes.

    try:
        report = classification_report(y_true, y_pred, target_names=disease_types, zero_division=0)
        print(report)
    except ValueError as e:
        print(f"Error generating classification report: {e}")
        print("This might be due to labels present in y_true or y_pred that are not in 'target_names'.")
        print(f"Unique true labels: {sorted(list(set(y_true)))}")
        print(f"Unique predicted labels: {sorted(list(set(y_pred)))}")
        print("Attempting report without explicit target_names (labels will be inferred):")
        try:
            report = classification_report(y_true, y_pred, zero_division=0)
            print(report)
        except Exception as e_inner:
            print(f"Could not generate report even with inferred labels: {e_inner}")


# --- Main Execution Block (Example Usage) ---
if __name__ == "__main__":
    # --- 1. Initialize models and settings once ---
    initialize_models()

    # --- 2. Choose mode: Single Prediction or Evaluation ---
    mode = input("Choose mode: (1) Single image prediction or (2) Evaluate on test dataset: ")

    if mode == '1':
        new_image_path = input("Enter the path to the fish image: ")
        if not new_image_path:
            print("No image path provided. Exiting.")
        else:
            print(f"\nPredicting on new image: {new_image_path}")
            predicted_disease, confidence, display_image = predict_disease(new_image_path)

            if predicted_disease and "Error" not in predicted_disease and "Failed" not in predicted_disease :
                print("\n--- Ensemble Prediction ---")
                print(f"-> Predicted Disease: {predicted_disease}")
                print(f"-> Final Confidence Score: {confidence:.4f}")

                if display_image is not None:
                    try:
                        plt.imshow(display_image.astype('uint8')) # Display non-normalized image
                        plt.title(f"Prediction: {predicted_disease} ({confidence:.2f})")
                        plt.axis('off')
                        plt.show()
                    except Exception as e:
                        print(f"Could not display image: {e}")
                else:
                    print("Could not display image (preprocessing might have failed for all displayable types).")
            else:
                print(f"Disease prediction failed or returned an error: {predicted_disease}")

    elif mode == '2':
        test_dataset_dir = input("Enter the path to the root directory of your test dataset: ")
        if not test_dataset_dir:
            print("No test dataset path provided. Exiting.")
        elif not os.path.isdir(test_dataset_dir):
            print(f"The path '{test_dataset_dir}' is not a valid directory. Exiting.")
        else:
            evaluate_and_generate_report(test_dataset_dir)
    else:
        print("Invalid mode selected. Exiting.")