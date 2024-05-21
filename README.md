# Automated-Number-Pate-Reader---Yolov8-with-Tesseract ( Real Time )

This project detects and reads license plates from a video. It uses a YOLO model to find license plates and Tesseract OCR to read the text on them. The detected license plate images and their text are saved to a folder and a CSV file. The project is built with Python using OpenCV for video handling, YOLO for detection, and Tesseract for text recognition.


## First Step - Train your YOLO model ( custom model )



1. **First download the files to train your YOLO model in the yolov8 format ,
   [Data to Train your model](https://universe.roboflow.com/samrat-sahoo/license-plates-f8vsn/dataset/5): so this includes:**
   
- train
- test
- valid
- **Note:** Use train images according to your spec 
2. **Train your model from Automated_Number_plate_reader.ipynb  file:**

## Features

- **License Plate Detection:** Automatically detects license plates within video footage using the YOLO model, ensuring accurate identification of vehicle registration information.

- **Text Recognition:**  Utilizes Tesseract OCR to extract text from detected license plates, enabling the system to interpret and process alphanumeric characters effectively.

- **Automated Saving:** Automatically saves both the cropped images of detected license plates and their corresponding text to a specified directory, streamlining data collection and storage for further analysis.

- **User Interaction:** Allows manual triggering of the saving process, giving users control over which license plates are processed and saved, enhancing flexibility and usability of the system.


## Local Development

Follow these instructions to set up and run this project on your local machine.

   **Note:** This project requires Python 3.10 or higher.

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/farzeencode/Automated-Number-Pate-Reader---Yolov8-with-Tesseract.git
   ```

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```



## Project Structure

- `app.py`: Main application script.
- `requirements.txt`: Python packages required for working of the app.
- `README.md`: Project documentation.

## Dependencies

- cv2
- cvzone
- math
- ultralytics
- pytesseract

## Acknowledgments

- [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki): Make sure that first install tesseract files
