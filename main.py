from ocr_handler import *



filename = input("Welcome to OCR video processer: ENTER FILENAME ")

if (os.path.isfile(filename)):  

    ocr_handler = OCR_HANDLER(filename,CV2_HELPER())
    ocr_handler.process_frames()
    ocr_handler.assemble_video()
    print ("OCR PROCESS FINISHED: OUT FILE =>" + ocr_handler.out_name)

else:
    print ("FILE NOT FOUND: BYE")