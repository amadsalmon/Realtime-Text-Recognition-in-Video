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
#Combine frames into video using opencv => Fix jerkynes


################################## CV2 FUNCTIONS ########################
class CV2_HELPER:
    # get grayscale image
    def get_grayscale(self,image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # noise removal
    def remove_noise(self,image):
        return cv2.medianBlur(image,5)
    
    #thresholding
    def thresholding(self,image):
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    #dilation
    def dilate(self,image):
        kernel = np.ones((5,5),np.uint8)
        return cv2.dilate(image, kernel, iterations = 1)

    #erosion
    def erode(self,image):
        kernel = np.ones((5,5),np.uint8)
        return cv2.erode(image, kernel, iterations = 1)

    #opening - erosion followed by dilation
    def opening(self,image):
        kernel = np.ones((5,5),np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    #canny edge detection
    def canny(self,image):
        return cv2.Canny(image, 100, 200)

    #skew correction
    def deskew(self,image):
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
    def match_template(self,image, template):
        return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 

################################## OCR PROCESSING ########################
class OCR_HANDLER:
    def __init__(self,video,cv2_helper):
        #The video's name with extension
        self.video=video
        self.cv2_helper=cv2_helper
        self.video_without_ext = self.video.split(".")[0]
        self.frames_folder = self.video_without_ext + '_frames'
        self.out_name = self.video_without_ext + '_boxes.avi'

    ########## EXTRACT FRAMES AND FIND WORDS #############
    def process_frames(self):
                
        frame_name = './' + self.frames_folder + '/' + self.video_without_ext + '_frame_'

        if not os.path.exists(self.frames_folder):
            os.makedirs(self.frames_folder)

        video = cv2.VideoCapture(self.video)        #Missing error code for when the video cannot be oppened
        self.fps = round(video.get(cv2.CAP_PROP_FPS))  # get the FPS of the video

        print("SAVING VIDEO AT",self.fps,"FPS")

        # get the list of duration spots to save
        saving_frames_durations = self.get_saving_frames_durations(video, self.fps) 

        idx = 0
        while True:
            is_read, frame = video.read()
            if not is_read:# break out of the loop if there are no frames to read
                break

            frame_duration = idx / self.fps
            try:
                # get the earliest duration to save
                closest_duration = saving_frames_durations[0]
            except IndexError:
                # the list is empty, all duration frames were saved
                break
            if frame_duration >= closest_duration:
                # if closest duration is less than or equals the frame duration, 
                # then save the frame

                output_name = frame_name + str(idx) + '.png'
                #frame=self.ocr_frame(frame)
                cv2.imwrite(output_name,frame)
                if idx % 10 == 0:
                    print ("Saving frame: ..."+output_name)
                # drop the duration spot from the list, since this duration spot is already saved
                try:
                    saving_frames_durations.pop(0)
                except IndexError:
                    pass
            # increment the frame count
            idx += 1

        print("Saved and processed",idx,"frames")
        video.release()

    def assemble_video(self):

        images = [img for img in os.listdir(self.frames_folder) if img.endswith(".png")] #Carefull with the order

        images = sorted(images, key=lambda x: float((x.split("_")[2])[:-4]))

        #Duplicate elements of the list to increase the frame count
        frame = cv2.imread(os.path.join(self.frames_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(self.out_name, 0, self.fps, (width,height))

        for image in images:
            video.write(cv2.imread(os.path.join(self.frames_folder, image)))

        video.release()
    
    def get_saving_frames_durations(self,video, saving_fps):
        """A function that returns the list of durations where to save the frames"""
        s = []
        # get the clip duration by dividing number of frames by the number of frames per second
        clip_duration = video.get(cv2.CAP_PROP_FRAME_COUNT) / video.get(cv2.CAP_PROP_FPS)
        # use np.arange() to make floating-point steps
        for i in np.arange(0, clip_duration, 1 / saving_fps):
            s.append(i)
        return s
    
    def ocr_frame(self,frame):
        #Pre-process the frame TODO play with these...
        gray = self.cv2_helper.get_grayscale(frame) 
        #thresh = self.cv2_helper.thresholding(gray)

        d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        n_boxes = len(d['text'])
        for i in range(n_boxes):
            if int(d['conf'][i]) > 60: #Confidence
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame


####################### MAIN ##############

ocr_handler = OCR_HANDLER('ocr2.mp4',CV2_HELPER())

ocr_handler.process_frames()
ocr_handler.assemble_video()

print ("OCR PROCESS FINISHED: OUT FILE =>" + ocr_handler.out_name)

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


