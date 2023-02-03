import cv2
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)


#names
classnames = []
classfile = 'coco.names'
with open(classfile,'rt') as f:
    classnames = f.read().rstrip('\n').split('\n')


config = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights = 'frozen_inference_graph.pb'

#model
net = cv2.dnn_DetectionModel(weights,config)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


    
while True:
    success,img = cap.read()
    classIds, con, bbox = net.detect(img,confThreshold=0.45)
    print(classIds,bbox)

    for classIds, confidence, box in zip(classIds.flatten(),con.flatten(),bbox):
        cv2.rectangle(img,box,color=(0,255,0),thickness=2)   #draw rectangle
        cv2.putText(img,classnames[classIds-1].upper(),(box[0]+10,box[1]+30),
            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2) #put text
    cv2.imshow("output",img)
    cv2.waitKey(1)




