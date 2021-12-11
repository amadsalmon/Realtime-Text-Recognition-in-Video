import cv2
import numpy as np
import pytesseract 
import os
#from matplotlib import pyplot as plot

#IDEA:
#Use opencv to break video into frames.
#Process frames to make text more readable using opencv
#Read out text and output it into terminal (pretty) using pytesseract
#For each frame add box surrounding text  => pending process frames (step2), to find position for the boxes (the same on the #original frame)
#Combine frames into video using opencv => Fix jerkyness

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 


if not os.path.exists('ocr1_frames'):
    os.makedirs('ocr1_frames')

video = cv2.VideoCapture('ocr1.mp4')

idx = 0
while video.isOpened():
    retval,frame = video.read()
    if not retval: #Last frame
        break

    output_name = './ocr1_frames/ocr1_frame_' + str(idx) + '.png'

    d = pytesseract.image_to_data(frame, output_type=pytesseract.Output.DICT)

    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 60:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite(output_name,frame)
    print ("Saving frame: ..."+output_name)

    idx +=1

video.release()

image_folder='ocr1_frames'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter('ocr1_boxes.avi', 0, 10, (width,height))  #3rd parameter => frames per second. it's a bit too jerky...

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

video.release()


#BOXES AROUND CHARACTERS
#boxes = pytesseract.image_to_boxes(frame) 
#    for b in boxes.splitlines():
#        b = b.split(' ')
#        frame = cv2.rectangle(frame, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)


#PROCESSING OF IMAGES TO MAKE THEM MORE READABLE

#image = cv2.imread('aurebesh.jpg')
#custom_config = r'--oem 3 --psm 6'
#pytesseract.image_to_string(image, config=custom_config)
#
#gray = get_grayscale(image)
##pytesseract.image_to_string(gray, config=custom_config)
#plot.imshow(gray)
#plot.show()
#
#thresh = thresholding(gray)
##pytesseract.image_to_string(thresh, config=custom_config)
#plot.imshow(thresh)
#plot.show()
#
#opening = opening(gray)
##pytesseract.image_to_string(opening, config=custom_config)
#plot.imshow(opening)
#plot.show()
#
#canny = canny(gray)
##pytesseract.image_to_string(canny, config=custom_config)
#plot.imshow(canny)
#plot.show()


