import cv2
import numpy as np
a = 0
b = 1
cap = cv2.VideoCapture(0)
firstFrame = None
text = "NotInTarget"
i = 0
count = 1
flag = 0
d = 0
sj = 0
tempx = 0
tempy = 0
tempr = 0
img11 = cv2.imread('circles.jpg',0)
retq,thresh3 = cv2.threshold(img11,200,255,1)
#cv2.imshow('fn',thresh3)
cnts11,hei = cv2.findContours(thresh3,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnt1 = cnts11[0]          
while True:
    

    grabbed, frame = cap.read()
	
    '''img3 = np.zeros((480,640,3), np.uint8)
    img3[100:250,450:640] = [255,255,255]
    img5 = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
    ret, img6 = cv2.threshold(img5,10, 255, cv2.THRESH_BINARY)
	
    img4 = cv2.bitwise_and(frame,frame,mask=img6)
    frame=imutils.resize(frame,width=500)
    gray=cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(21,21),0)'''
    edgeFrame=cv2.Canny(frame,100,100)
    
    circles = cv2.HoughCircles(edgeFrame,cv2.HOUGH_GRADIENT,3,550)
    if (cv2.waitKey(1) & 0xff == ord('a')):
        a = 1
          
    if a==1:
        if circles is not None:
    
            circles = np.uint16(np.around(circles))

            for i in circles[0, :]:
                if((i[2]<100)&(i[2]>50)):
                    cv2.circle(frame,(i[0],i[1]),i[2],(0,0,255),10)
                    circleDetected = 1
                    x = i[0]
                    y = i[1]
                    r = i[2]
                    img = np.zeros((480,640,3), np.uint8)
                    #img[i[0]-2*i[2]:i[0]+2*i[2],i[1]-2*i[2]:i[1]+2*i[2]] = [255,255,255]
                    cv2.circle(img,(x,y),i[2]+30,(255,255,255),-1)
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    ret, mask1 = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)
                    mask_inv = cv2.bitwise_not(mask1)
                    a = 2
                    print(x,y,r)
                else :
                    continue
        print ("in loop")        
    cv2.imshow("output", frame)
           

    if a==2:
        if circles is not None:
    
            circles = np.uint16(np.around(circles))
            temp = 0
            for i in circles[0, :]:
                tempx = i[0]
                tempy = i[1]
                tempr = i[2]
                '''if((i[0]>(x-20)) & (i[0]<(x+20))&(i[1]>(y-20))&(i[1]<(y+20)) &(i[2]>50) & (i[2]<100)):
                    cv2.circle(frame,(i[0],i[1]),i[2],(0,0,255),10)
                    circleDetected=1
                    img = np.zeros((480,640,3), np.uint8)
                    #img[i[0]-2*i[2]:i[0]+2*i[2],i[1]-2*i[2]:i[1]+2*i[2]] = [255,255,255]
                    cv2.circle(img,(x,y),2*i[2],(255,255,255),-1)
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    ret, mask1 = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)
                    mask_inv = cv2.bitwise_not(mask1)'''
                #print (i[0],i[1],i[2],x)
                temp = i[0]
                img1 = cv2.bitwise_and(frame,frame,mask = mask_inv)
                #cv2.imshow("ROI", img1)
        
            if not sj:
                if(abs(temp-x)>40):
                    count = count - 1
                else:
                    d = d + 1
                    if(d == 5):
                        sj = 1
            if count == 0:
                a = 1
                count = 1
            if (sj==1)&(flag==0):
                x = tempx
                y = tempy
                r = tempr
                flag = 1
        cv2.imshow("output", frame)

        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,(21,21),0)
        #cv2.imshow('graythresh',gray)
        text = "NotInTarget"
	    
        if firstFrame is None:
            firstFrame = gray
	    #cv2.imshow('df',firstFrame)
            continue

        frameDelta = cv2.absdiff(firstFrame,gray)
        thresh = cv2.threshold(frameDelta,25,255,cv2.THRESH_BINARY)[1]

        thresh = cv2.dilate(thresh,None,iterations=2)
        (cnts,h) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        #ret = cv2.matchShapes(cnt1,cnts[0],1,0.0)
        #cv2.matchShapes(cnt1,cnts[0],1,0.0)
        for c in cnts:
            ret = cv2.matchShapes(cnt1,cnts[0],2,0.0)
            
            if(ret<0.007):
                print (ret)
                
                if (cv2.contourArea(c)>1500):
                    continue
                (x1, y1, w, h) = cv2.boundingRect(c)
                print(x1-x,y1-y)
                if(((abs((x1+w/2)-x))<r)&((abs((y1+h/2)-y))<r)):
                    cv2.putText(img1, "success".format(text), (100, 100),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                cv2.rectangle(img1, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                text = "Target"
				

        cv2.putText(img1, "Room Status: {}".format(text), (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow("Frame", img1)
        #cv2.imshow("Thresh", thresh)
        #cv2.imshow("Frame Delta", frameDelta)
        #key = cv2.waitKey(1) & 0xFF
	#cv2.imshow('frame',img4)
	#cv2.imwrite('frame%d.png'%i,frame)
        i=i+1
        firstFrame=gray
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
		
cap.release()
cv2.destroyAllWindows()
