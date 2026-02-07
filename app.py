import face_recognition
import cv2
import os
import gspread
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

app = Flask(__name__)

# --- CONFIG ---
DATASET_PATH = 'dataset'
SHEET_ID = "1kMbG_96D552CcXQcijLVH8ggFQxN8qLRbZdL1N-6oTc"
attendance_done = False

# Google Sheets Setup
sheet = None
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

@app.route('/')
def index():
    global attendance_done
    attendance_done = False
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global attendance_done, sheet
    if attendance_done:
        return jsonify({"status": "Done"})

    try:
        data = request.json['image'].split(',')[1]
        img_data = base64.b64decode(data)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        faces_info = []
        res_name = "Unknown"
        status = "Scanning"
        found_known = False

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.45)
            name = "Unknown"
            color = "red"

            if True in matches:
                idx = matches.index(True)
                name = known_names[idx]
                color = "green"
                res_name = name
                found_known = True

            faces_info.append({"box": [top, right, bottom, left], "name": name, "color": color})

        if found_known and not attendance_done:
            now = datetime.now()
            t_str = now.strftime('%H:%M:%S')
            d_str = now.strftime('%d-%m-%Y')
            if sheet:
                sheet.append_row([res_name, t_str, d_str])
            attendance_done = True
            return jsonify({"status": "Success", "name": res_name, "time": t_str, "date": d_str, "faces": faces_info})

        return jsonify({"status": "Scanning", "name": res_name, "faces": faces_info})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"status": "Error", "message": str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)