import streamlit as st
import cv2
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.utils import to_categorical
from PIL import Image

st.title("🧠 ANN Face Recognition System")

# Ensure dataset folder exists
os.makedirs("dataset", exist_ok=True)

# =====================================================
# 1️⃣ COLLECT FACE IMAGES
# =====================================================
st.header("Step 1: Collect Face Images")

person_name = st.text_input("Enter Person Name")
camera_image = st.camera_input("Capture Face")

if camera_image is not None and person_name.strip() != "":

    save_path = f"dataset/{person_name}"
    os.makedirs(save_path, exist_ok=True)

    image = Image.open(camera_image)
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.warning("⚠ No face detected. Try again.")
    else:
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (64, 64))

            count = len(os.listdir(save_path))
            cv2.imwrite(f"{save_path}/{count}.jpg", face)

            st.success(f"📸 Image {count+1} saved successfully!")

# =====================================================
# 2️⃣ TRAIN MODEL
# =====================================================
st.header("Step 2: Train ANN Model")

if st.button("Train Model"):

    data = []
    labels = []
    path = "dataset"

    persons = os.listdir(path)

    # Safety: Need at least 2 people
    if len(persons) < 2:
        st.error("❌ Please collect images for at least 2 different persons.")
        st.stop()

    for person in persons:
        person_path = os.path.join(path, person)
        for img in os.listdir(person_path):
            img_path = os.path.join(person_path, img)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (64, 64))
            image = image / 255.0
            data.append(image)
            labels.append(person)

    data = np.array(data)
    labels = np.array(labels)

    unique_labels = np.unique(labels)
    label_dict = {label: i for i, label in enumerate(unique_labels)}

    encoded_labels = np.array([label_dict[label] for label in labels])
    encoded_labels = to_categorical(encoded_labels)

    X_train, X_test, y_train, y_test = train_test_split(
        data, encoded_labels, test_size=0.2, random_state=42
    )

    # Modern Sequential Model (No Warning Version)
    model = Sequential([
        Input(shape=(64, 64)),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(len(unique_labels), activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(X_train, y_train, epochs=40, batch_size=4, verbose=0)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    model.save("face_model.h5")

    with open("label_dict.pkl", "wb") as f:
        pickle.dump(label_dict, f)

    st.success(f"✅ Model Trained Successfully! Accuracy: {accuracy:.2f}")

# =====================================================
# 3️⃣ FACE RECOGNITION
# =====================================================
st.header("Step 3: Face Recognition")

uploaded_image = st.camera_input("Capture Image for Recognition")

if uploaded_image is not None:

    # Safety check
    if not os.path.exists("face_model.h5"):
        st.stop()

    if not os.path.exists("label_dict.pkl"):
        st.error("❌ Label file not found. Please train the model first.")
        st.stop()

    model = load_model("face_model.h5")

    with open("label_dict.pkl", "rb") as f:
        label_dict = pickle.load(f)

    label_dict = {v: k for k, v in label_dict.items()}

    image = Image.open(uploaded_image)
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.warning("⚠ No face detected.")
    else:
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (64, 64))
            face = face / 255.0
            face = face.reshape(1, 64, 64)

            prediction = model.predict(face, verbose=0)
            class_index = np.argmax(prediction)
            confidence = np.max(prediction)

            name = label_dict[class_index]

            cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(
                image,
                f"{name} ({confidence*100:.1f}%)",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )

        st.image(image)