import json, time
from custom_detector import detect
from flask import Flask, Response
from threading import Thread

app = Flask(__name__)

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

def frame_generator():
    while True:
        time.sleep(0.5)
        if FRAME:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + FRAME + b'\r\n\r\n'

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

if __name__ == '__main__':
    app.run(debug=True)