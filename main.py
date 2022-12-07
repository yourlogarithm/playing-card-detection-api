import json, time
from io import BytesIO
from custom_detector import detect
from flask import Flask, Response, send_file
from flask_cors import CORS
from threading import Thread

app = Flask(__name__)
cors = CORS(app)


DETECTING = False
FRAME = b''
DETECTIONS = []

def make_detections():
    global DETECTING, FRAME, DETECTIONS
    DETECTING = True
    for frame, detections in detect():
        if frame: 
            FRAME = frame
        DETECTIONS = detections

def get_frame():
    return b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + FRAME + b'\r\n\r\n'

def frame_generator():
    while True:
        time.sleep(0.5)
        if FRAME:
            yield get_frame()

def detections_generator():
    if DETECTIONS:
        return json.dumps([detection.to_dict() for detection in DETECTIONS])

@app.route('/video_feed')
def video_feed():
    if not DETECTING: Thread(target=make_detections).start()
    return Response(frame_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def detections():
    if not DETECTING: Thread(target=make_detections).start()
    return Response(detections_generator(), mimetype='application/json')

@app.route('/capture')
def capture():
    if FRAME:
        img_io = BytesIO(FRAME)
        img_io.seek(0)
        return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)