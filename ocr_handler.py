import cv2
import numpy as np
import pytesseract
import os
import math
from pathlib import Path

INPUT_DIR = "./input/"
OUTPUT_DIR = "./output/"


# TODO check this preprocessing steps https://towardsdatascience.com/pre-processing-in-ocr-fc231c6035a7

################################## CV2 FUNCTIONS ########################
class CV2_HELPER:

    # Returns a binary image using an adaptative threshold
    def binarization_adaptative_threshold(self, image):
        # 11 => size of a pixel neighborhood that is used to calculate a threshold value for the pixel
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # skew correction to align image with horizontal
    def deskew(self, image):
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
        return rotated, angle

    # smoothen the image by removing small dots/patches which have high intensity than the rest of the image
    def remove_noise(self, image):
        return cv2.medianBlur(image, 5)

    # to make the width of strokes uniform, we have to perform Thinning and Skeletonization
    def erode(self, image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.erode(image, kernel, iterations=1)

    # get grayscale image
    def get_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # dilation
    def dilate(self, image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(image, kernel, iterations=1)

    # opening - erosion followed by dilation
    def opening(self, image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # canny edge detection
    def canny(self, image):
        return cv2.Canny(image, 100, 200)

    # template matching
    def match_template(self, image, template):
        return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

    ################################## OCR PROCESSING ########################

class BOXES_HELPER():
    
    def get_organized_tesseract_dictionary(self,tesseract_dictionary):
        res = {}
        n_boxes = len(tesseract_dictionary['level'])

        # Organize blocks
        res['blocks'] = {}
        for i in range(n_boxes):
            if tesseract_dictionary['level'][i] == 2:
                res['blocks'][tesseract_dictionary['block_num'][i]] = {
                    'left': tesseract_dictionary['left'][i],
                    'top': tesseract_dictionary['top'][i],
                    'width': tesseract_dictionary['width'][i],
                    'height': tesseract_dictionary['height'][i],
                    'paragraphs': {}
                }

        # Organize paragraphs
        for i in range(n_boxes):
            if tesseract_dictionary['level'][i] == 3:
                res['blocks'][tesseract_dictionary['block_num'][i]]['paragraphs'][tesseract_dictionary['par_num'][i]]   = {
                    'left': tesseract_dictionary['left'][i],
                    'top': tesseract_dictionary['top'][i],
                    'width': tesseract_dictionary['width'][i],
                    'height': tesseract_dictionary['height'][i],
                    'lines': {}
                }

        # Organize lines
        for i in range(n_boxes):
            if tesseract_dictionary['level'][i] == 4:
                res['blocks'][tesseract_dictionary['block_num'][i]]['paragraphs'][tesseract_dictionary['par_num'][
                    i]]['lines'][tesseract_dictionary['line_num'][i]] = {
                    'left': tesseract_dictionary['left'][i],
                    'top': tesseract_dictionary['top'][i],
                    'width': tesseract_dictionary['width'][i],
                    'height': tesseract_dictionary['height'][i],
                    'words': {}
                }

        # Organize words
        for i in range(n_boxes):
            if tesseract_dictionary['level'][i] == 5:
                res['blocks'][tesseract_dictionary['block_num'][i]]['paragraphs'][
                    tesseract_dictionary['par_num'][
                        i]]['lines'][tesseract_dictionary['line_num'][i]]['words'][tesseract_dictionary['word_num'][i]  ] \
                    = {
                    'left': tesseract_dictionary['left'][i],
                    'top': tesseract_dictionary['top'][i],
                    'width': tesseract_dictionary['width'][i],
                    'height': tesseract_dictionary['height'][i],
                    'text': tesseract_dictionary['text'][i],
                    'conf': float(tesseract_dictionary['conf'][i]),
                }

        return res


    def get_lines_with_words(self,organized_tesseract_dictionary):
        res = []
        for block in organized_tesseract_dictionary['blocks'].values():
            for paragraph in block['paragraphs'].values():
                for line in paragraph['lines'].values():
                    if 'words' in line and len(line['words']) > 0:
                        currentLineText = ''
                        for word in line['words'].values():
                            if word['conf'] > 60.0 and not word['text'].isspace():
                                currentLineText += word['text'] + ' '
                        if currentLineText != '':
                            res.append({'text': currentLineText, 'left': line['left'], 'top': line['top'], 'width':     line[
                                'width'], 'height': line[
                                'height']})

        return res


    def show_boxes_lines(self,d, frame):
        text_vertical_margin = 12
        organized_tesseract_dictionary = self.get_organized_tesseract_dictionary(d)
        lines_with_words = self.get_lines_with_words(organized_tesseract_dictionary)
        # print(lines_with_words)
        for line in lines_with_words:
            x = line['left']
            y = line['top']
            h = line['height']
            w = line['width']
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            frame = cv2.putText(frame,
                                text=line['text'],
                                org=(x, y - text_vertical_margin),
                                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                fontScale=1,
                                color=(0, 255, 0),
                                thickness=2)
        return frame

    def show_boxes_words(self,d,frame):
        text_vertical_margin = 12
        n_boxes = len(d['level'])
        for i in range(n_boxes):
            if (int(float(d['conf'][i])) > 80) and not (d['text'][i].isspace()):  # Words
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                frame = cv2.putText(frame, text=d['text'][i], org=(x, y - text_vertical_margin),
                                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                     fontScale=1,
                                     color=(0, 255, 0), thickness=2)
        return frame



class OCR_HANDLER:

    def __init__(self, video_filepath, cv2_helper, ocr_type="WORDS"):
        # The video_filepath's name with extension
        self.video_filepath = video_filepath
        self.cv2_helper = cv2_helper
        self.ocr_type= ocr_type
        self.boxes_helper = BOXES_HELPER()
        self.video_name = Path(self.video_filepath).stem
        self.frames_folder = OUTPUT_DIR + 'temp/' + self.video_name + '_frames'
        self._fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change to 'MP4V' if this doesn't work on your OS.
        self.out_extension = '.avi'
        self.out_name = self.video_name + '_boxes' + self.out_extension

    ########## EXTRACT FRAMES AND FIND WORDS #############
    def process_frames(self):

        frame_name = './' + self.frames_folder + '/' + self.video_name + '_frame_'

        if not os.path.exists(self.frames_folder):
            os.makedirs(self.frames_folder)

        video = cv2.VideoCapture(
            self.video_filepath)  # TODO Missing error code for when the video_filepath cannot be oppened
        self.fps = round(video.get(cv2.CAP_PROP_FPS))  # get the FPS of the video_filepath
        frames_durations, frame_count = self.get_saving_frames_durations(video, self.fps)  # list of point to save

        print("SAVING VIDEO:", frame_count, "FRAMES AT", self.fps, "FPS")

        idx = 0
        print(":", end='', flush=True)
        while True:
            print("=", end='', flush=True)
            is_read, frame = video.read()
            if not is_read:  # break out of the loop if there are no frames to read
                break
            frame_duration = idx / self.fps
            try:
                # get the earliest duration to save
                closest_duration = frames_durations[0]
            except IndexError:
                # the list is empty, all duration frames were saved
                break
            if frame_duration >= closest_duration:
                # if closest duration is less than or equals the frame duration, then save the frame
                output_name = frame_name + str(idx) + '.png'
                frame = self.ocr_frame(frame)
                cv2.imwrite(output_name, frame)

                if (idx % 10 == 0) and (idx > 0):
                    print(">")
                    print("Saving frame: ..." + output_name)
                    print(":", end='', flush=True)
                # drop the duration spot from the list, since this duration spot is already saved
                try:
                    frames_durations.pop(0)
                except IndexError:
                    pass
            # increment the frame count
            idx += 1
        if (idx - 1 % 10 != 0):
            print(">")
        print("\nSaved and processed", idx, "frames")
        video.release()

    def assemble_video(self):

        print("ASSEMBLING NEW VIDEO")

        images = [img for img in os.listdir(self.frames_folder) if img.endswith(".png")]  # Careful with the order
        images = sorted(images, key=lambda x: float((x.split("_")[-1])[:-4]))

        frame = cv2.imread(os.path.join(self.frames_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(OUTPUT_DIR + self.out_name, 0, self.fps, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(self.frames_folder, image)))

        video.release()

        # When finished, delete all frames stored temporarily on disk.
        for f in os.listdir(self.frames_folder):
            if not f.endswith(".png"):
                continue
            try:
                os.remove(os.path.join(self.frames_folder, f))
            except OSError as e:
                print("Error: %s : %s" % (self.frames_folder, e.strerror))

        # Then delete the directory that contained the frames.
        try:
            os.rmdir(self.frames_folder)
        except OSError as e:
            print("Error: %s : %s" % (self.frames_folder, e.strerror))

    def get_saving_frames_durations(self, video, saving_fps):
        """A function that returns the list of durations where to save the frames"""
        s = []
        # get the clip duration by dividing number of frames by the number of frames per second
        clip_duration = video.get(cv2.CAP_PROP_FRAME_COUNT) / video.get(cv2.CAP_PROP_FPS)
        # use np.arange() to make floating-point steps
        for i in np.arange(0, clip_duration, 1 / saving_fps):
            s.append(i)
        return s, video.get(cv2.CAP_PROP_FRAME_COUNT)

    def ocr_frame(self, frame):
        # Pre-process the frame TODO play with preprocessing and segmentation.

        im, d = self.compute_best_preprocess(self.cv2_helper.get_grayscale(frame))

        if (self.ocr_type == "LINES"):
            frame = self.boxes_helper.show_boxes_lines(d, frame)
        else:
            frame = self.boxes_helper.show_boxes_words(d, frame)

        return frame

    def compute_best_preprocess(self, frame):

        #img = self.cv2_helper.binarization_adaptative_threshold(frame)  # Binarization
        #img = self.cv2_helper.remove_noise(img)
        #img = self.cv2_helper.erode(img)
        #d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

        def f(count,mean):
            return 10*count + mean

        best_f=0
        best_opt=0
        best_im = frame
        best_d = None
        options = [["binarization"],["binarization","remove_noise"],["binarization","remove_noise","erode"]]

        for idx, opt in enumerate(options):
            #Apply preprocess
            im=frame
            if "binarization" in opt:
                im = self.cv2_helper.binarization_adaptative_threshold(im)
            if "deskew" in opt:
                im = self.cv2_helper.deskew(im)
            if "remove_noise" in opt:
                im = self.cv2_helper.remove_noise(im)
            if "erode" in opt:
                im = self.cv2_helper.erode(im)

            #Compute mean conf:
            d = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT)
            confs = [ int(d['conf'][i]) for i in range(len(d['text'])) if not(d['text'][i].isspace())]
            confs = [i for i in confs if i > 60]

            mean_conf = np.asarray(confs).mean() if len(confs) > 0 else 0

            #print(len(confs),mean_conf,f(len(confs),mean_conf))

            if (f(len(confs),mean_conf) > best_f):
                best_im = im
                best_d = d
                best_f = f(len(confs),mean_conf)

        return best_im,best_d



