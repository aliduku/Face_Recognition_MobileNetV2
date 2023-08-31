import cv2
import numpy as np
from keras.models import load_model
# Load the trained model
model = load_model('weights.h5')

# Load the label names
target_names = [
    'Alejandro Toledo', 'Alvaro Uribe', 'Amelie Mauresmo', 'Andre Agassi',
    'Angelina Jolie', 'Ariel Sharon', 'Arnold Schwarzenegger',
    'Atal Bihari Vajpayee', 'Bill Clinton', 'Carlos Menem', 'Colin Powell',
    'David Beckham', 'Donald Rumsfeld', 'George Robertson', 'George W Bush',
    'Gerhard Schroeder', 'Gloria Macapagal Arroyo', 'Gray Davis',
    'Guillermo Coria', 'Hamid Karzai', 'Hans Blix', 'Hugo Chavez', 'Igor Ivanov',
    'Jack Straw', 'Jacques Chirac', 'Jean Chretien', 'Jennifer Aniston',
    'Jennifer Capriati', 'Jennifer Lopez', 'Jeremy Greenstock', 'Jiang Zemin',
    'John Ashcroft', 'John Negroponte', 'Jose Maria Aznar',
    'Juan Carlos Ferrero', 'Junichiro Koizumi', 'Kofi Annan', 'Laura Bush',
    'Lindsay Davenport', 'Lleyton Hewitt', 'Luiz Inacio Lula da Silva',
    'Mahmoud Abbas', 'Megawati Sukarnoputri', 'Michael Bloomberg', 'Naomi Watts',
    'Nestor Kirchner', 'Paul Bremer', 'Pete Sampras', 'Recep Tayyip Erdogan',
    'Ricardo Lagos', 'Roh Moo-hyun', 'Rudolph Giuliani', 'Saddam Hussein',
    'Serena Williams', 'Silvio Berlusconi', 'Tiger Woods', 'Tom Daschle',
    'Tom Ridge', 'Tony Blair', 'Vicente Fox', 'Vladimir Putin', 'Winona Ryder'
]

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open the webcam
cap = cv2.VideoCapture(0)

# Set the confidence threshold for predictions
confidence_threshold = 0.8

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region
        face = frame[y:y+h, x:x+w]

        # Resize the face to match the model's input size
        face_resized = cv2.resize(face, (160, 160))

        # Preprocess the face image for the model
        face_resized = np.expand_dims(face_resized, axis=0) / 255.0

        # Perform face recognition using the model
        embedding = model.predict(face_resized)
        predicted_class = np.argmax(embedding)
        confidence = np.max(embedding)

        # Check if confidence is above the threshold
        if confidence > confidence_threshold and predicted_class < len(target_names):
            predicted_name = target_names[predicted_class]
        else:
            predicted_name = 'Unknown'

        # Draw bounding box around the face and label it with confidence
        label_text = f'{predicted_name} ({confidence:.2f})'
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()