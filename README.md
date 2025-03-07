#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
camera_capture.py

Description:
1. The function capture_single_image(): Opens the camera, captures a single image, and saves it in 16-bit format.
2. The function capture_continuously(): Continuously captures images at a specified frame rate and saves them as 16-bit .npy files.
3. Several helper functions (analyze_frame, show_16bit_with_matplotlib, etc.) are also provided for optional use.

Usage:
- Open this script in an IDE such as PyCharm and run it. In main(), call the desired functions.
- Alternatively, import capture_single_image or capture_continuously from this module in another script.
"""

import os
import time
import numpy as np
import pylablib as pll
from pylablib.devices import Thorlabs
import matplotlib.pyplot as plt
import cv2

# Initialize and list available backend resources (e.g., serial ports)
pll.list_backend_resources("serial")


def show_16bit_with_matplotlib(frame_16u):
    """
    Display a 16-bit image using matplotlib.
    
    Args:
        frame_16u: A uint16 image with shape (height, width) or (height, width, channels)
    """
    plt.figure()
    plt.imshow(frame_16u, cmap='gray', vmin=0, vmax=65535)
    plt.colorbar(label='Pixel value')
    plt.title("16-bit image (Matplotlib)")
    plt.show()


def analyze_frame(frame):
    """
    Print basic statistics of the provided frame.
    """
    print("Shape:", frame.shape)
    print("Dtype:", frame.dtype)
    print("Min:", frame.min())
    print("Max:", frame.max())
    print("Mean:", frame.mean())


def capture_single_image(name='background', exposure_time=0.04):
    """
    Description:
      Opens the camera, sets the exposure, captures a single image, saves it as a 16-bit .npy file 
      (optionally also as PNG), and then closes the camera.
    
    Usage:
      from camera_capture import capture_single_image
      capture_single_image()
    
    Args:
        name: Base name for the saved image file.
        exposure_time: Exposure time in seconds.
    """
    # 1. Open the camera
    cam = Thorlabs.ThorlabsTLCamera()

    # 2. (Optional) Set ROI (Region of Interest)
    # width, height = 1440, 1080
    # cam.set_roi(0, width, 0, height, hbin=1, vbin=1)

    # 3. Set the exposure time (in seconds, e.g., 0.01 for 10 ms)
    cam.set_exposure(exposure_time)

    # 4. Start acquisition (nframes=1 means capture one frame)
    cam.setup_acquisition(nframes=1)
    cam.start_acquisition()

    try:
        # 5. Wait for and read the image
        cam.wait_for_frame(timeout=10.0)
        frame = cam.read_newest_image()
        analyze_frame(frame)

        # 6. (Optional) Visualize the image (uncomment if needed)
        # show_16bit_with_matplotlib(frame)

        # 7. Save the image as a .npy file (16-bit raw data)
        np.save(name + '.npy', frame)
        print(f"[INFO] 16-bit image saved as: {name}.npy")
        '''
        # 8. Also save as a 16-bit PNG (OpenCV supports writing 16-bit images)
        cv2.imwrite('single_frame_16bit.png', frame)
        print("[INFO] 16-bit image saved as: single_frame_16bit.png")
        '''

    finally:
        # 9. Close the camera
        cam.close()
        print("[INFO] Camera closed.")


def capture_continuously(func, model, bg_gray=[0],
                         min_area=2500,
                         morph_kernel_size=3,
                         threshold_val=4,
                         exposure_time=0.05, frame_num=10):
    """
    Continuously capture images and process each frame using the provided function.
    The processing function (e.g., classification) returns a result image (with bounding boxes/labels)
    that is displayed in real-time.
    
    Args:
        func: The processing function to be applied on each frame.
        model: Model or additional parameter required by the processing function.
        bg_gray: Background gray image for subtraction.
        min_area: Minimum area for a contour to be considered valid.
        morph_kernel_size: Kernel size for morphological operations.
        threshold_val: Threshold value for binarization.
        exposure_time: Camera exposure time in seconds.
        frame_num: Desired number of frames per second.
    """
    cam = Thorlabs.ThorlabsTLCamera()
    cam.set_exposure(exposure_time)

    cam.setup_acquisition(nframes=frame_num)
    cam.start_acquisition()

    frame_count = 0
    last_time = time.time()
    last_frame_time = last_time
    file_index = 0

    print(f"[INFO] Starting continuous capture at {frame_num} frames per second.")

    try:
        while True:
            now = time.time()

            # Capture a frame when enough time has elapsed (>= 1/frame_num)
            if (now - last_frame_time) >= 1.0 / frame_num:
                frame = cam.read_newest_image()
                if frame is None:
                    continue

                # Call the processing function (e.g., classification)
                flag, result_img = func(
                    str(file_index),
                    model,
                    bg_gray,
                    frame,  # original 16-bit grayscale image
                    min_area,
                    morph_kernel_size,
                    threshold_val
                )

                # If flag == 0, then stop capturing
                if flag == 0:
                    print("[INFO] Processing function returned 0, breaking the loop.")
                    break

                # Display the result image in real-time (result_img is an 8-bit BGR image)
                cv2.imshow("Live Preview", result_img)

                # Process key press for exit (ESC key = 27)
                if cv2.waitKey(1) & 0xFF == 27:
                    print("[INFO] ESC key pressed, exiting loop.")
                    break

                # Update counters and timestamps
                file_index += 1
                frame_count += 1
                last_frame_time = now

            # Calculate and display actual FPS every 1 second
            if (now - last_time) >= 1.0:
                fps = frame_count / (now - last_time)
                print(f"[INFO] Actual FPS: {fps:.2f}")
                frame_count = 0
                last_time = now

    except KeyboardInterrupt:
        print("[INFO] Caught Ctrl+C, closing camera.")
    finally:
        cam.close()
        cv2.destroyAllWindows()  # Close all windows
        print("[INFO] Camera closed, script ended.")


def i(base_name, bg_gray, sample, min_area, morph_kernel_size, threshold_val):
    answer_bg = input("Do you want to capture a background image? (y/n): ").strip().lower()
    print(answer_bg)
    return 0


def detect_rice_by_subtraction(bg_gray, sample_gray,
                               min_area=1000,
                               morph_kernel_size=2,
                               threshold_val=4):
    """
    Detect rice in an image using pixel subtraction.
    
    Args:
        bg_gray: Background gray image (16-bit).
        sample_gray: Current sample frame in grayscale.
        min_area: Minimum contour area to consider.
        morph_kernel_size: Kernel size for morphological operations.
        threshold_val: Threshold value for binarization.
    
    Returns:
        final_image: Color image (BGR) with valid contours drawn in red.
        fg_mask: Binary mask from thresholding.
        valid_contours: List of valid contours after filtering.
    """
    # Compute the absolute difference between background and sample images
    diff = cv2.absdiff(bg_gray, sample_gray)

    # Apply threshold to create a binary mask
    _, fg_mask = cv2.threshold(diff, threshold_val, 255, cv2.THRESH_BINARY)

    # Use morphological operations to remove noise and fill holes
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find contours in the binary mask
    fg_mask = fg_mask.astype(np.uint8)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours and draw valid contours on the image
    # Convert grayscale image to BGR for drawing colored contours
    final_image = cv2.cvtColor(sample_gray, cv2.COLOR_GRAY2BGR)
    tmp = final_image.astype(np.float32)
    # Normalize to [0,1]
    tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
    # Scale back to [0,255] and convert to uint8
    final_image = (tmp * 255).astype(np.uint8)
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        valid_contours.append(cnt)
        # Draw contour in red (BGR: (0, 0, 255))
        cv2.drawContours(final_image, [cnt], -1, (0, 0, 255), 2)

    return final_image, fg_mask, valid_contours


import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']   # Set Chinese font to 'SimHei' (or other)
matplotlib.rcParams['axes.unicode_minus'] = False     # Ensure minus signs are displayed correctly


def label_contours(image, contours):
    """
    Label each contour with a unique index on a single-channel grayscale image.
    
    Args:
        image: Grayscale image.
        contours: List of contours.
        
    Returns:
        The image with drawn labels.
    """
    for i, cnt in enumerate(contours):
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        # Draw the label in white (255)
        cv2.putText(
            image, str(i), (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, 255, 2, cv2.LINE_AA
        )
    return image


def get_training_set(base_name, nothing, bg_gray, sample, min_area, morph_kernel_size, threshold_val):
    """
    1. Perform subtraction to get contours (grayscale).
    2. Label the image with indices.
    3. Interactively ask the user to confirm whether each detected rice contour is normal or should be removed.
    4. Save the results.
    
    Returns:
        A tuple (flag, dummy) where flag==0 indicates end, flag==1 to continue.
    """
    # 1. Detect rice contours using grayscale subtraction
    final_image, fg_mask, valid_contours = detect_rice_by_subtraction(
        bg_gray=bg_gray,
        sample_gray=sample,
        min_area=min_area,
        morph_kernel_size=morph_kernel_size,
        threshold_val=threshold_val
    )

    # 2. Label contours for user identification
    labeled_image = label_contours(final_image, valid_contours)

    # 3. Display the labeled image in an OpenCV window
    cv2.namedWindow('Labeling Window', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Labeling Window', 800, 600)
    cv2.imshow('Labeling Window', labeled_image)

    normal_flags = []
    filtered_contours = []

    for i, cnt in enumerate(valid_contours):
        # Process window events to prevent freezing
        cv2.waitKey(1)
        user_input = input(f"Is rice at index {i} normal? (y/n/e): ").lower().strip()

        if user_input in ["y", "yes"]:
            normal_flags.append(True)
            filtered_contours.append(cnt)
        elif user_input in ["n", "no"]:
            normal_flags.append(False)
            filtered_contours.append(cnt)
        elif user_input in ["e", "error"]:
            print(f"Contour at index {i} has been removed.")
        else:
            print(f"Input '{user_input}' is invalid, contour removed.")

    cv2.destroyAllWindows()

    # 4. Save contours and labels (only saving the kept contours)
    conts_as_arrays = [cnt.reshape(-1, 2) for cnt in filtered_contours]
    data_dict = {
        "is_normal": normal_flags,
        "contours": conts_as_arrays,
        "data": sample  # Optionally, save the original image as well
    }

    # Save path example: "train_set/<base_name>_setted.npy"
    save_name = base_name + "_setted"
    save_folder = "train_set"
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, save_name)

    np.save(save_path, data_dict, allow_pickle=True)
    print(f"Data saved to {save_path}")
    ans = input("Finish? (y/n): ").strip().lower()
    if ans == 'y':
        return 0, 0
    else:
        return 1, 0


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


def extract_gray_features(image_gray_uint16, contour):
    """
    Extract [mean, std] features from a given contour on a 16-bit or 8-bit grayscale image.
    Modify this function if you use different features during training.
    
    Args:
        image_gray_uint16: Grayscale image (uint16 or uint8)
        contour: Contour array
        
    Returns:
        A list containing [gray_mean, gray_std].
    """
    if contour.ndim == 2:
        contour = contour.reshape(-1, 1, 2)

    # Create a mask for the contour
    h, w = image_gray_uint16.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [contour.astype(np.int32)], -1, color=255, thickness=-1)

    if mask.sum() == 0:
        # Return default features if the contour area is 0
        return [0.0, 0.0]

    mean_val, std_val = cv2.meanStdDev(image_gray_uint16, mask=mask)
    gray_mean = float(mean_val[0][0])
    gray_std = float(std_val[0][0])
    return [gray_mean, gray_std]


def build_dataset(train_set_dir):
    """
    Traverse all .npy files in the train_set_dir folder,
    read the data_dict from each file, and parse out is_normal (labels), contours, and the original image (grayscale, uint16).
    Then, extract [mean, std] features for each contour and store them in X and y.
    
    Returns:
        X: Feature matrix (N, 2)
        y: Label array (N,)
    """
    X = []
    y = []

    # Traverse .npy files in the directory
    for filename in os.listdir(train_set_dir):
        if not filename.endswith(".npy"):
            continue  # Skip non-npy files

        npy_path = os.path.join(train_set_dir, filename)
        data_dict = np.load(npy_path, allow_pickle=True).item()

        # data_dict should contain "is_normal", "contours", "data"
        # data is a grayscale uint16 image with shape (H, W)
        image_gray_uint16 = data_dict["data"]
        contours = data_dict["contours"]
        is_normal_list = data_dict["is_normal"]

        # Extract features for each contour
        for cnt, normal_flag in zip(contours, is_normal_list):
            features = extract_gray_features(image_gray_uint16, cnt)
            X.append(features)
            # Convert True/False to 1/0 for binary classification
            label = 1 if normal_flag else 0
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def train_model(train_set_dir):
    """
    Train a Random Forest model using the dataset built from the training set.
    """
    # 1. Build the training dataset
    X, y = build_dataset(train_set_dir)
    print("Feature matrix X shape:", X.shape)  # Expected shape (N, 2)
    print("Label array y shape:", y.shape)       # Expected shape (N,)

    # 2. Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Train a Random Forest classifier
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42
    )
    rf.fit(X_train, y_train)

    # 4. Evaluate the model on the test set
    y_pred = rf.predict(X_test)
    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Abnormal Rice", "Normal Rice"]))

    # 5. (Optional) Save the trained model
    model_save_path = "rice_rf_model.pkl"
    joblib.dump(rf, model_save_path)
    print(f"Model saved to: {model_save_path}")


def classification(base_name, model, bg_gray, sample,
                   min_area, morph_kernel_size, threshold_val):
    """
    Perform foreground detection and Random Forest classification on a single frame (sample),
    then draw the classification results (green/red boxes) on the image.
    
    Returns:
        flag: 0 or 1 indicating whether to continue capturing (1 = continue, 0 = stop).
        result_img: 8-bit BGR image with detection/classification results for real-time display.
    """
    # 1. Perform subtraction to get contours
    final_image, fg_mask, valid_contours = detect_rice_by_subtraction(
        bg_gray=bg_gray,
        sample_gray=sample,
        min_area=min_area,
        morph_kernel_size=morph_kernel_size,
        threshold_val=threshold_val
    )
    # final_image is an 8-bit BGR image (converted from the original grayscale) with red contours

    # 2. Extract features for each contour and predict using the trained model
    pred_results = []
    for contour in valid_contours:
        features = extract_gray_features(sample, contour)
        features = np.array(features, dtype=np.float32).reshape(1, -1)
        pred_label = model.predict(features)[0]  # 0 (abnormal) or 1 (normal)
        pred_results.append(pred_label)

    # 3. Draw bounding boxes and text labels based on prediction results
    result_img = final_image.copy()
    for contour, pred_label in zip(valid_contours, pred_results):
        # Prediction: 0 = abnormal (red), 1 = normal (green)
        color = (0, 255, 0) if pred_label == 1 else (0, 0, 255)
        label_text = "Normal" if pred_label == 1 else "Abnormal"

        # Get bounding rectangle coordinates
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(result_img, label_text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 4. (Optional) Save the result image to file
    out_dir = "results_set"
    os.makedirs(out_dir, exist_ok=True)
    out_filename = base_name + "_inference.png"
    out_path = os.path.join(out_dir, out_filename)
    cv2.imwrite(out_path, result_img)
    
    # 5. (Optional) Save the prediction labels for record keeping
    pred_data = {
        "valid_contours": valid_contours,
        "pred_labels": pred_results
    }
    out_npy_path = os.path.join(out_dir, base_name + "_pred.npy")
    np.save(out_npy_path, pred_data, allow_pickle=True)
    
    # 6. Return flag (1 to continue) and the result image
    return 1, result_img


if __name__ == '__main__':
    exposure_time = 0.05
    frame_num = int(0.5 / exposure_time)

    # -----------------------
    # Step 1: Capture Background Image
    # -----------------------
    answer_bg = input("Do you want to capture a background image? (y/n): ").strip().lower()
    if answer_bg == 'y':
        print("Capturing background image...")
        capture_single_image(name='background', exposure_time=exposure_time)
        bg_gray_path = "background.npy"
    elif answer_bg == 'n':
        # Use existing background image if not capturing a new one
        bg_gray_path = "background.npy"
    else:
        bg_gray_path = "background.npy"
    bg_gray = np.load("background.npy")
    print(f"Background loaded from: {bg_gray_path}")

    # -----------------------
    # Step 2: Set Processing Parameters
    # -----------------------
    answer_param = input("Do you want to adjust detection parameters? (y/n): ").strip().lower()

    if answer_param == 'n':
        # Directly load parameters from detect_parameters.npy
        if not os.path.exists('detect_parameters.npy'):
            print("File detect_parameters.npy does not exist. Please adjust parameters first.")
    else:
        print("Starting parameter tuning process...")
        # Capture a sample image for tuning
        capture_single_image(name='find_parameters', exposure_time=exposure_time)
        sample_path = 'find_parameters.npy'
        sample_gray = np.load(sample_path)

        # Use matplotlib to display the detection result
        while True:
            try:
                min_area = int(input("Enter min_area (number): "))
                morph_kernel_size = int(input("Enter morph_kernel_size (number): "))
                threshold_val = int(input("Enter threshold_val (number): "))
            except ValueError:
                print("Invalid input, please try again.")
                continue

            final_img, fg_mask, valid_contours = detect_rice_by_subtraction(
                bg_gray=bg_gray,
                sample_gray=sample_gray,
                min_area=min_area,
                morph_kernel_size=morph_kernel_size,
                threshold_val=threshold_val
            )

            # Convert BGR image to RGB for proper display in matplotlib
            show_final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)

            plt.figure(figsize=(8, 6))
            plt.imshow(show_final_img)
            plt.title(f"min_area={min_area}, morph_kernel_size={morph_kernel_size}, threshold_val={threshold_val}")
            plt.show()

            ans_satisfied = input("Are you satisfied with the current detection result? (y/n): ").strip().lower()
            if ans_satisfied == 'y':
                print("Saving parameters to detect_parameters.npy.")
                params_dict = {
                    'min_area': min_area,
                    'morph_kernel_size': morph_kernel_size,
                    'threshold_val': threshold_val
                }
                np.save('detect_parameters.npy', params_dict)
                break
            else:
                print("Not satisfied, please enter the parameters again.")

    params_dict = np.load('detect_parameters.npy', allow_pickle=True).item()
    min_area = params_dict.get('min_area', 1000)
    morph_kernel_size = params_dict.get('morph_kernel_size', 2)
    threshold_val = params_dict.get('threshold_val', 4)
    print("Detection parameters loaded from detect_parameters.npy:")
    print(f"min_area = {min_area}, morph_kernel_size = {morph_kernel_size}, threshold_val = {threshold_val}")

    # -----------------------
    # Step 3: Set Up Training Set
    # -----------------------
    answer_st = input("Do you want to set up the training set? (y/n): ").strip().lower()
    if answer_st == 'y':
        capture_continuously(get_training_set, 0, bg_gray=bg_gray,
                              min_area=min_area,
                              morph_kernel_size=morph_kernel_size,
                              threshold_val=threshold_val,
                              exposure_time=exposure_time, frame_num=frame_num)
        train_set_dir = 'train_set'
    else:
        train_set_dir = 'train_set'
    print(f"Training set loaded from: {train_set_dir}")

    # -----------------------
    # Step 4: Train Model
    # -----------------------
    answer_st = input("Do you want to train the model? (y/n): ").strip().lower()
    if answer_st == 'y':
        train_model(train_set_dir)
        model_path = "rice_rf_model.pkl"
    else:
        model_path = "rice_rf_model.pkl"
    print(f"Model ready at: {model_path}")
    model = joblib.load(model_path)
    print("Model loaded.")

    capture_continuously(classification, model, bg_gray=bg_gray,
                          min_area=min_area,
                          morph_kernel_size=morph_kernel_size,
                          threshold_val=threshold_val,
                          exposure_time=exposure_time, frame_num=frame_num)











# -*- coding: utf-8 -*-
"""
camera_capture_color.py

Example: Complete color camera script. The workflow is similar to the "black & white camera" script and can be executed step-by-step in PyCharm.

Please modify according to your actual camera SDK/API and requirements.
If you are not using ThorlabsTLCamera, please replace the camera initialization and image capture functions.
"""

import os
import time
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt

# If you are actually using pylablib + Thorlabs:
try:
    import pylablib as pll
    from pylablib.devices import Thorlabs
except ImportError:
    print("[WARN] Could not import pylablib / Thorlabs SDK; some demonstration code may not run properly.")


# ========== 1) Capture a Single (Color) Image ==========
def capture_single_image_color(name='color_background', exposure_time=0.04):
    """
    Open the color camera, capture an image, and save it as a .npy file.
    Assumes the output is 16-bit BGR (H, W, 3).
    """
    # --- Open the camera (example) ---
    cam = Thorlabs.ThorlabsTLCamera()

    # --- Set the exposure time (in seconds) ---
    cam.set_exposure(exposure_time)

    # --- Start acquisition ---
    cam.setup_acquisition(nframes=1)
    cam.start_acquisition()

    try:
        # Wait for and read the frame
        cam.wait_for_frame(timeout=10.0)
        frame_color = cam.read_newest_image()
        # Print frame info
        print("Shape:", frame_color.shape, "Dtype:", frame_color.dtype)

        # Save as .npy
        np.save(name + '.npy', frame_color)
        print(f"[INFO] Color image saved as {name}.npy")

    finally:
        cam.close()
        print("[INFO] Color camera closed.")


# ========== 2) Continuous Capture (Color) ==========
def capture_continuously_color(
    func,            # Processing function, e.g.: func(base_name, model, bg_color, frame, ...)
    model,
    bg_color=None,   # Background image (if using difference)
    min_area=500,
    morph_kernel_size=3,
    threshold_val=30,
    exposure_time=0.05,
    frame_num=10
):
    """
    Continuously capture color images at frame_num FPS.
    Each image is passed to func() for processing and the result is displayed.
    If func returns (0, _), the loop will be terminated.
    """
    cam = Thorlabs.ThorlabsTLCamera()
    cam.set_exposure(exposure_time)
    cam.setup_acquisition(nframes=frame_num)
    cam.start_acquisition()

    frame_count = 0
    last_time = time.time()
    last_frame_time = last_time
    file_index = 0

    print(f"[INFO] Starting continuous capture of color images with target FPS: {frame_num}.")

    try:
        while True:
            now = time.time()
            if (now - last_frame_time) >= 1.0 / frame_num:
                frame_color = cam.read_newest_image()
                if frame_color is None:
                    continue

                # Call the processing function
                flag, result_img = func(
                    str(file_index),
                    model,
                    bg_color,
                    frame_color,
                    min_area,
                    morph_kernel_size,
                    threshold_val
                )

                if flag == 0:
                    print("[INFO] Function returned 0, terminating loop.")
                    break

                # Display the processed result
                cv2.imshow("Live Preview (Color)", result_img)
                if cv2.waitKey(1) & 0xFF == 27:
                    print("[INFO] ESC key pressed, exiting loop.")
                    break

                file_index += 1
                frame_count += 1
                last_frame_time = now

            # Calculate actual FPS every second
            if (now - last_time) >= 1.0:
                fps = frame_count / (now - last_time)
                print(f"[INFO] Actual FPS: {fps:.2f}")
                frame_count = 0
                last_time = now

    except KeyboardInterrupt:
        print("[INFO] Interrupted by Ctrl+C.")
    finally:
        cam.close()
        cv2.destroyAllWindows()
        print("[INFO] Camera closed, ending process.")


# ========== 3) Difference Detection (Color Version) ==========
def detect_rice_by_subtraction_color(
    bg_color,
    sample_color,
    min_area=500,
    morph_kernel_size=3,
    threshold_val=30
):
    """
    Simple color difference: perform absdiff on the three channels, convert to grayscale,
    apply thresholding and morphological operations.
    Returns (final_image, fg_mask, valid_contours)
      - final_image: contours drawn on sample_color (8-bit BGR)
    """
    # 1) Absolute difference
    diff_bgr = cv2.absdiff(bg_color, sample_color)

    # 2) Convert to grayscale
    diff_gray = cv2.cvtColor(diff_bgr, cv2.COLOR_BGR2GRAY)

    # 3) Thresholding
    _, fg_mask = cv2.threshold(diff_gray, threshold_val, 255, cv2.THRESH_BINARY)

    # 4) Morphological operations
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 5) Find contours
    contours, _ = cv2.findContours(fg_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 6) Filter small areas and draw contours in red on the image
    final_image = sample_color.copy().astype(np.uint8)
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        valid_contours.append(cnt)
        cv2.drawContours(final_image, [cnt], -1, (0, 0, 255), 2)

    return final_image, fg_mask, valid_contours


# ========== 4) Interactive Training Set Labeling (Color) ==========
def label_contours(image, contours):
    """
    Label each contour on a BGR image.
    """
    for i, cnt in enumerate(contours):
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        cv2.putText(image, str(i), (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return image

def get_training_set_color(
    base_name,      # Used for constructing the save filename
    unused_model,   # Not used here
    bg_color,
    sample_color,
    min_area,
    morph_kernel_size,
    threshold_val
):
    """
    Interactive calibration using color images. Can be called during capture.
    1) Color difference -> Contour extraction
    2) Label contours
    3) Terminal interaction (y/n/e)
    4) Save to train_set/..._setted.npy
    """
    final_image, fg_mask, valid_contours = detect_rice_by_subtraction_color(
        bg_color=bg_color,
        sample_color=sample_color,
        min_area=min_area,
        morph_kernel_size=morph_kernel_size,
        threshold_val=threshold_val
    )

    labeled_img = label_contours(final_image, valid_contours)

    # Display window
    cv2.namedWindow('Labeling Window (Color)', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Labeling Window (Color)', 800, 600)
    cv2.imshow('Labeling Window (Color)', labeled_img)

    normal_flags = []
    filtered_contours = []

    for i, cnt in enumerate(valid_contours):
        cv2.waitKey(1)
        user_input = input(f"Is rice at contour {i} [Normal]? (y/n/e): ").strip().lower()
        if user_input in ["y", "yes"]:
            normal_flags.append(True)
            filtered_contours.append(cnt)
        elif user_input in ["n", "no"]:
            normal_flags.append(False)
            filtered_contours.append(cnt)
        else:
            print(f"Contour {i} has been deleted/ignored.")

    cv2.destroyAllWindows()

    # Save training data
    conts_as_arrays = [c.reshape(-1, 2) for c in filtered_contours]
    data_dict = {
        "is_normal": normal_flags,
        "contours": conts_as_arrays,
        "data": sample_color  # Save the original image
    }

    os.makedirs("train_set", exist_ok=True)
    save_path = os.path.join("train_set", base_name + "_setted.npy")
    np.save(save_path, data_dict, allow_pickle=True)
    print(f"[INFO] Interactive labeling completed and saved to: {save_path}")

    # Return (flag, None); in this example, if the user wants to exit early
    ans = input("End labeling? (y=end/n=continue): ").strip().lower()
    if ans == "y":
        return 0, 0  # Signal to stop
    else:
        return 1, 0  # Continue


# ========== 5) Feature Extraction & Model Training ==========
def extract_color_features_16bit(image_bgr_uint16, contour):
    """
    Extract features: [R_mean, G_mean, B_mean, R_std, G_std, B_std]
    """
    if contour.ndim == 2:
        contour = contour.reshape(-1, 1, 2)
    x, y, w, h = cv2.boundingRect(contour)
    roi = image_bgr_uint16[y:y+h, x:x+w]
    contour_shifted = contour.copy()
    contour_shifted[..., 0] -= x
    contour_shifted[..., 1] -= y

    mask_local = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask_local, [contour_shifted.astype(np.int32)], -1, 255, -1)

    mean_val, std_val = cv2.meanStdDev(roi, mask=mask_local)
    if mask_local.sum() == 0:
        return [0, 0, 0, 0, 0, 0]
    B_mean, G_mean, R_mean = mean_val.flatten()
    B_std, G_std, R_std = std_val.flatten()
    return [R_mean, G_mean, B_mean, R_std, G_std, B_std]

def build_dataset(train_set_dir):
    """
    Traverse the *_setted.npy files in train_set_dir and generate (X, y).
    """
    X, y = [], []
    for fname in os.listdir(train_set_dir):
        if not fname.endswith(".npy"):
            continue
        path = os.path.join(train_set_dir, fname)
        data_dict = np.load(path, allow_pickle=True).item()
        image_bgr_uint16 = data_dict["data"]
        contours = data_dict["contours"]
        normal_flags = data_dict["is_normal"]
        for cnt, flag in zip(contours, normal_flags):
            feat = extract_color_features_16bit(image_bgr_uint16, cnt)
            X.append(feat)
            y.append(1 if flag else 0)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def train_model(train_set_dir):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    X, y = build_dataset(train_set_dir)
    print("X shape:", X.shape, "y shape:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Abnormal", "Normal"]))

    joblib.dump(rf, "rice_rf_model.pkl")
    print("[INFO] Model saved as rice_rf_model.pkl")


# ========== 6) Real-time Classification Inference (Color) ==========
def classification_color(
    base_name,
    model,         # Trained model
    bg_color,      # Background image (16-bit BGR)
    sample_color,  # Current frame (16-bit BGR)
    min_area,
    morph_kernel_size,
    threshold_val
):
    """
    Similar to the black & white script: perform foreground detection -> feature extraction -> classification ->
    draw bounding boxes and save the image.
    """
    final_img, fg_mask, valid_contours = detect_rice_by_subtraction_color(
        bg_color=bg_color,
        sample_color=sample_color,
        min_area=min_area,
        morph_kernel_size=morph_kernel_size,
        threshold_val=threshold_val
    )

    # Predict labels
    pred_results = []
    for cnt in valid_contours:
        feats = extract_color_features_16bit(sample_color, cnt)
        feats = np.array(feats, dtype=np.float32).reshape(1, -1)
        label = model.predict(feats)[0]  # 0=Abnormal / 1=Normal
        pred_results.append(label)

    # Draw bounding boxes and text on the image
    result_img = final_img.copy()
    for cnt, label in zip(valid_contours, pred_results):
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        text = "Normal" if label == 1 else "Abnormal"
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(result_img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(result_img, text, (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Optionally save the result image
    out_dir = "results_inference"
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, base_name + "_color_infer.png")
    cv2.imwrite(out_png, result_img)

    # Return (continue_flag, display_image)
    return 1, result_img


# ========== 7) Main Process (Similar to the Black & White Script) ==========
if __name__ == '__main__':

    # 1) Capture background image
    ans_bg = input("Capture COLOR background? (y/n): ").strip().lower()
    if ans_bg == "y":
        capture_single_image_color(name="color_background", exposure_time=0.02)
        bg_color_path = "color_background.npy"
    else:
        bg_color_path = "color_background.npy"

    if not os.path.exists(bg_color_path):
        raise FileNotFoundError("Color background not found. Please capture or specify an existing background file.")
    bg_color = np.load(bg_color_path)
    print(f"[INFO] Loaded color background: {bg_color_path}")

    # 2) Check if you want to adjust detection parameters
    ans_param = input("Adjust detection parameters? (y/n): ").strip().lower()
    if ans_param == "y":
        # Capture an image for parameter tuning
        capture_single_image_color(name="find_parameters_color", exposure_time=0.02)
        sample_path = "find_parameters_color.npy"
        sample_color = np.load(sample_path)

        while True:
            try:
                min_area = int(input("Enter min_area: "))
                morph_kernel_size = int(input("Enter morph_kernel_size: "))
                threshold_val = int(input("Enter threshold_val: "))
            except ValueError:
                print("[WARN] Invalid input format.")
                continue

            # Perform detection
            final_img, fg_mask, valid_contours = detect_rice_by_subtraction_color(
                bg_color=bg_color,
                sample_color=sample_color,
                min_area=min_area,
                morph_kernel_size=morph_kernel_size,
                threshold_val=threshold_val
            )

            # Display the image using matplotlib
            show_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
            plt.figure()
            plt.imshow(show_img)
            plt.title(f"min_area={min_area}, morph={morph_kernel_size}, thresh={threshold_val}")
            plt.show()

            ans_ok = input("Are you satisfied with these parameters? (y/n): ").strip().lower()
            if ans_ok == "y":
                print("Parameters saved to detect_parameters_color.npy")
                params_dict = {
                    "min_area": min_area,
                    "morph_kernel_size": morph_kernel_size,
                    "threshold_val": threshold_val
                }
                np.save("detect_parameters_color.npy", params_dict)
                break
    else:
        if not os.path.exists("detect_parameters_color.npy"):
            print("[WARN] detect_parameters_color.npy not found. Using default parameters.")
            min_area = 500
            morph_kernel_size = 3
            threshold_val = 30
        else:
            params = np.load("detect_parameters_color.npy", allow_pickle=True).item()
            min_area = params.get("min_area", 500)
            morph_kernel_size = params.get("morph_kernel_size", 3)
            threshold_val = params.get("threshold_val", 30)

    print(f"[INFO] Using parameters: min_area={min_area}, morph_kernel_size={morph_kernel_size}, threshold_val={threshold_val}")

    # 3) Collect training set data
    ans_ts = input("Set up a training set? (y/n): ").strip().lower()
    if ans_ts == "y":
        exposure_time = 0.02
        frame_num = int(0.5 / exposure_time)  # Example calculation
        capture_continuously_color(
            get_training_set_color,
            model=0,
            bg_color=bg_color,
            min_area=min_area,
            morph_kernel_size=morph_kernel_size,
            threshold_val=threshold_val,
            exposure_time=exposure_time,
            frame_num=frame_num
        )

    # 4) Train the model
    ans_tr = input("Train the model? (y/n): ").strip().lower()
    if ans_tr == "y":
        train_model(train_set_dir="train_set")

    model_path = "rice_rf_model.pkl"
    if not os.path.exists(model_path):
        print("[WARN] Model file not found. Classification will not be available. Please train a model or place an existing one.")
    else:
        # 5) Real-time classification
        ans_run = input("Start real-time COLOR classification? (y/n): ").strip().lower()
        if ans_run == "y":
            model = joblib.load(model_path)
            print(f"[INFO] Loaded model: {model_path}")

            exposure_time = 0.02
            frame_num = int(0.5 / exposure_time)
            capture_continuously_color(
                classification_color,
                model=model,
                bg_color=bg_color,
                min_area=min_area,
                morph_kernel_size=morph_kernel_size,
                threshold_val=threshold_val,
                exposure_time=exposure_time,
                frame_num=frame_num
            )

    print("[INFO] Process complete.")

