import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('skin_lesion_model.h5')

# Define the lesion type mapping
lesion_type_dict = {
    0: 'Melanocytic nevi',
    1: 'Melanoma',
    2: 'Benign keratosis-like lesions',
    3: 'Basal cell carcinoma',
    4: 'Actinic keratoses',
    5: 'Vascular lesions',
    6: 'Dermatofibroma'
}

# Function to capture image from webcam
def capture_image_from_camera():
    cap = cv2.VideoCapture(1)  # 0 is usually the default camera
    while True:
        ret, frame = cap.read()  # Capture frame-by-frame
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow("Press 's' to capture image", frame)  # Display the frame

        # Press 's' to capture and save the image
        if cv2.waitKey(1) & 0xFF == ord('s'):
            captured_img = frame
            cv2.imwrite("captured_image.png", captured_img)  # Save image (optional)
            break

    cap.release()  # Release the camera
    cv2.destroyAllWindows()
    return captured_img

def preprocess_image(img, size):
    img = cv2.resize(img, (size, size))
    img = img.astype(np.float32) / 255.0 
    img = np.expand_dims(img, axis=0)  
    return img

# Function to classify the image
def classify_image(model, preprocessed_image):
    predictions = model.predict(preprocessed_image)
    predicted_class_idx = np.argmax(predictions, axis=1)
    predicted_class = lesion_type_dict[predicted_class_idx[0]]
    return predicted_class

# Step 1: Capture the image from the camera
captured_image = capture_image_from_camera()

# Step 2: Preprocess the captured image
preprocessed_image = preprocess_image(captured_image, 128)

# Step 3: Classify the image using the pre-trained model
predicted_class = classify_image(model, preprocessed_image)

# Step 4: Output the result
print(f"The predicted class is: {predicted_class}")