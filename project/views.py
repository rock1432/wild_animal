from django.shortcuts import render
from ultralytics import YOLO
import cv2
import math
from django.http import StreamingHttpResponse
import requests


model = YOLO("project/model/best.pt")

def send_db(predictions):
    pred = str(predictions)
    print('predicted', pred)
    url = 'http://iotcloud22.in/3351_animal/post_value.php'
    data = {"value1": pred}
    response = requests.post(url, data=data)
    print("HTTP Response:", response.status_code)


def home(request):
    return render(request, 'home.html')

def wild_page(request):
    return render(request, 'detect.html')

def detection(request):
    return StreamingHttpResponse(start_live(request), content_type="multipart/x-mixed-replace;boundary=frame")

def start_live(request):
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    classNames = ['', 'elephant', 'leopard', 'boar', 'tiger']



    while True:
        success, img = cap.read()
        results = model(img, stream=True)

        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                confidence = math.ceil((box.conf[0]*100))/100
                print("Confidence --->",confidence)
                if confidence > 0.70:
                    cls = int(box.cls[0])
                    # print("class : ",cls)
                    name=classNames[cls]
                    print("Class name -->", name)

                    org = [x1, y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (0, 0, 255)
                    thickness = 2
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
                    if classNames[cls] == '':
                        print('##########2')
                        send_db('')
                    else:
                        print('##########3')
                        send_db(classNames[cls])    
                print('##########1')        
                send_db('')
        # cv2.imshow('Webcam', img)
        _, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        # Yield the frame as bytes for the StreamingHttpResponse
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return render(request, 'detect.html')    
