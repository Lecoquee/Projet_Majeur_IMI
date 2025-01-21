import cv2 as cv
x = 2
if x == 1:
    for i in range(16):
        cap = cv.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera {i} trouvée et accessible")
            cap.release()
        else:
            print(f"Port {i} NADA")

if x == 2:

    cap1 = cv.VideoCapture(8)
    cap2 = cv.VideoCapture(0)
    #cap3 = cv.VideoCapture(8)


    while True:
        ret1, frame1 = cap1.read() 
        ret2, frame2 = cap2.read()
        #ret3, frame3 = cap3.read()

        cv.imshow('Camera 1', frame1)
        cv.imshow('Camera 2', frame2)
        #cv.imshow('Camera 3', frame3)

        if cv.waitKey(1) == 27:
            break