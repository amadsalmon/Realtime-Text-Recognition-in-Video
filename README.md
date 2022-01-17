# Text Recognition in Video


The intelligent analysis of video data is currently in wide demand because a video is a major source of sensory data in our lives. Text is a prominent and direct source of information in video.  
This project is aimed at harnessing the power of computer vision to implement text recognition in video.
It is carried out as part of the Computer Vision course held at the Sapienza University of Rome.

## Authors

- Amad Salmon (@amadsalmon)
- Arturo Calvera Tonin (@Arturo-00)


## Installing the required dependencies

### OpenCV-Python

```
pip install opencv-python
```
### Tesseract Engine

See https://tesseract-ocr.github.io/tessdoc/Installation.html

### Python-Tesseract

Python-tesseract is a python wrapper for Google's Tesseract-OCR.  
See https://pypi.org/project/pytesseract/.

```
pip install pytesseract
```

## Execution

Run `main.py`.  
Provide the path for a video file. We suggest short `.mp4` video files.  
Choose between word-by-word or line-by-line recognition.

Output file will be provided in the `output/` folder.

## References

- X. Yin, Z. Zuo, S. Tian and C. Liu, "Text Detection, Tracking and Recognition in Video: A Comprehensive Survey," in IEEE Transactions on Image Processing, vol. 25, no. 6, pp. 2752-2773, June 2016, doi: 10.1109/TIP.2016.2554321. [[link](https://ieeexplore.ieee.org/document/7452620)]