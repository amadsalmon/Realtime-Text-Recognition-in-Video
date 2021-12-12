# Import required packages
from cv2 import cv2
import pytesseract

# Mention the installed location of Tesseract-OCR in your system
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

LIST = ['director', 'directed', 'written', 'writer',
        'producer', 'produced', 'production', 'lead']

path = 'video_sample1.mov'  # input('Enter path to file or hls manifest: ')

vidcap = cv2.VideoCapture(path)
success, frame = vidcap.read()
framenbr = 0

while success:
    success, frame = vidcap.read()
    if not success:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    data = pytesseract.image_to_string(gray, lang='eng', config='--psm 6').lower()
    print(data + '\n-- Frame number: ' + str(framenbr))
    framenbr += 1
