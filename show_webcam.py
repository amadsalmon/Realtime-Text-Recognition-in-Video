import cv2


def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)

    print('Press "Esc" to exit.')
    while True:
        ret_val, img = cam.read()

        if mirror:
            img = cv2.flip(img, 1)
        cv2.imshow('Camera video capture', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


def main():
    show_webcam(mirror=True)


if __name__ == '__main__':
    main()