import cv2
import cvzone
import math
from ultralytics import YOLO
import os
import csv
import pytesseract

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path
video=r"E:\Yolo\demo_video.mp4"

# Initialize video capture
cap = cv2.VideoCapture(video)  # Using webcam for live feed

# Load YOLO model
model = YOLO('best.pt')
classnames = ['license-plate', 'vehicle']

# Ensure directories exist for saving plates and CSV file
plates_folder = 'captured_plates'
csv_file = 'license_plates.csv'
if not os.path.exists(plates_folder):
    os.makedirs(plates_folder)

# Initialize CSV file and serial number
serial_no = 1
if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Serial No', 'Plate Text'])

# Initialize plate counter
plate_counter = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame = cv2.resize(frame, (1080, 720))
        results = model(frame)
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_detect = int(box.cls[0])
                class_name = classnames[class_detect]
                
                if confidence > 0.5 and class_name == 'license-plate':
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cvzone.putTextRect(frame, f'{class_name} {confidence:.2f}', [x1, y1 - 10], scale=1, thickness=2)
                    
                    # Crop the detected plate
                    img_crop = frame[y1:y2, x1:x2]

                    key = cv2.waitKey(1)
                    if key == ord('s'):
                        # Perform OCR on the cropped plate image
                        plate_text = pytesseract.image_to_string(img_crop, config='--psm 6')  # PSM 6 is for treating the image as a single block of text
                        
                        # Save the plate text and serial number to CSV file
                        with open(csv_file, 'a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([serial_no, plate_text.strip()])

                        serial_no += 1

                        # Save the cropped plate image
                        cv2.imwrite(os.path.join(plates_folder, f"plate_{plate_counter}.jpg"), img_crop)
                        cv2.rectangle(frame, (0, 200), (500, 300), (0, 255, 0), cv2.FILLED)
                        cv2.putText(frame, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                        cv2.imshow("Result", frame)
                        cv2.waitKey(500)
                        plate_counter += 1

                        # Show the extracted text
                        print("Extracted text from number plate:", plate_text)
                        cv2.imshow("Cropped Plate", img_crop)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('t'):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
