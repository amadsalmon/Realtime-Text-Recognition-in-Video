from ocr_handler import *

print("Welcome to OCR video processer.")
filename = input("ENTER FILENAME: ")

if os.path.isfile(filename):
    ocr_handler = OCR_HANDLER(filename, CV2_HELPER())
    ocr_handler.process_frames()
    ocr_handler.assemble_video()
    print("OCR PROCESS FINISHED: OUTPUT FILE => " + ocr_handler.out_name)

else:
    print("FILE NOT FOUND: BYE")
