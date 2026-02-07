import face_recognition
import cv2
import os
import gspread
import numpy as np
from flask import Flask, render_template, Response, jsonify
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

app = Flask(__name__)

# --- CONFIG ---
DATASET_PATH = 'dataset'
SHEET_ID = "1kMbG_96D552CcXQcijLVH8ggFQxN8qLRbZdL1N-6oTc"
attendance_done = False
current_user = "Scanning..."
last_time = "--:--:--"
last_date = "--/--/----"

# Google Sheets Setup
try:
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(SHEET_ID).sheet1
    print("✅ Google Sheets Connected")
except Exception as e:
    print(f"⚠️ Sheets Error: {e}")

# Pre-load Dataset
known_encodings = []
known_names = []

print("🚀 Loading Dataset...")
if os.path.exists(DATASET_PATH):
    for name in os.listdir(DATASET_PATH):
        person_dir = os.path.join(DATASET_PATH, name)
        if not os.path.isdir(person_dir): continue
        for img_name in os.listdir(person_dir):
            img = face_recognition.load_image_file(os.path.join(person_dir, img_name))
            enc = face_recognition.face_encodings(img)
            if enc:
                known_encodings.append(enc[0])
                known_names.append(name)

def gen_frames():
    global attendance_done, current_user, last_time, last_date
    cap = cv2.VideoCapture(0)
    
    while True:
        if attendance_done:
            break

        success, frame = cap.read()
        if not success: break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.4)
            if True in matches:
                idx = matches.index(True)
                name = known_names[idx]
                
                if not attendance_done:
                    current_user = name
                    now = datetime.now()
                    last_time = now.strftime('%H:%M:%S')
                    last_date = now.strftime('%d-%m-%Y')
                    sheet.append_row([name, last_time, last_date])
                    attendance_done = True 
                    break # Loop se bahar

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release() # CAMERA HARDWARE BAND HOGA
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    if attendance_done: return "Stopped"
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_data')
def get_data():
    return jsonify({"name": current_user, "time": last_time, "date": last_date, "status": "Done" if attendance_done else "Scanning"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)