import cv2
import numpy as np

#REQUIRED INITILIZATIONS
class_names = []
width, height, channel = 800, 600,3
bounding_box_list = []
confidence_list = []
class_ids_list = []
class_names = []


#LOADING CLASS NAMES INTO class_names FROM coco.names FILE
with open("coco.names","rt") as file:
    class_names = file.read().split("\n")
print(class_names)

#LOADING THE NETWORK
net = cv2.dnn.readNetFromDarknet("yolov3.cfg","yolov3.weights")

#LOADING THE CAMERA
cap = cv2.VideoCapture(0)

while True:
    flag, frame = cap.read()
    frame = cv2.resize(frame,(800,600))
    # print("Dimension = ", frame.shape)

    #CONVERTING EACH FRAME INTO BLOB TYPE
    blob = cv2.dnn.blobFromImage(frame,1/255,(416,416),[0,0,0],1,crop=False)

    #FEEDING THE BLOB INTO THE IMAGE
    net.setInput(blob)

    #FINDING THE NAME OF OUTPUT LAYERS
    layer_names = net.getLayerNames() # GIVES THE NAME OF ALL THE LAYERS OF THE NETWORK
    out_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

    #GETTING OUTPUTS FROM THE THREE OUTPUT LAYERS
    outputs = net.forward(out_layers)
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)

                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                bounding_box_list.append([x,y,w,h])
                confidence_list.append(float(confidence))
                class_ids_list.append(class_id)

    #PERFORMING NON MAXIMUM SUPRESSION
    indices = cv2.dnn.NMSBoxes(bounding_box_list,confidence_list,0.5,0.5)
    for index in indices:
        i = index[0]
        box = bounding_box_list[i]
        x,y,w,h = box[0],box[1], box[2], box[3]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(frame, class_names[class_ids_list[i]]
                    ,(x+10,y+20),cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0),3)
        cv2.putText(frame, f'{int(confidence_list[i]*100)}%'
                    ,(x+150,y+20),cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0),3)

    cv2.imshow("YOLO video frame", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()