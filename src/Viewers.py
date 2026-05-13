import cv2 as cv
import numpy as np

def nothing(x):
    pass

def HsvViewer():
    win = "HSV Viewer"
    cv.namedWindow(win)

    cv.createTrackbar("H", win, 0, 179, nothing)
    cv.createTrackbar("S", win, 255, 255, nothing)
    cv.createTrackbar("V", win, 255, 255, nothing)

    while True:
        h = cv.getTrackbarPos("H", win)
        s = cv.getTrackbarPos("S", win)
        v = cv.getTrackbarPos("V", win)

        hsv_color = np.uint8([[[h, s, v]]])
        bgr_color = cv.cvtColor(hsv_color, cv.COLOR_HSV2BGR)[0][0]

        canvas = np.zeros((500, 700, 3), dtype=np.uint8)
        canvas[:] = (40, 40, 40)

        # color preview box
        cv.rectangle(
            canvas,
            (50, 80),
            (350, 380),
            tuple(int(x) for x in bgr_color),
            -1
        )

        cv.rectangle(canvas, (50, 80), (350, 380), (255, 255, 255), 2)

        # text info
        b, g, r = bgr_color

        cv.putText(canvas, "HSV Viewer", (50, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv.putText(canvas, f"H: {h}", (420, 120),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv.putText(canvas, f"S: {s}", (420, 170),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv.putText(canvas, f"V: {v}", (420, 220),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv.putText(canvas, f"BGR: ({b}, {g}, {r})", (420, 280),
                   cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv.putText(canvas, "OpenCV HSV range:", (420, 340),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv.putText(canvas, "H: 0~179", (420, 370),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv.putText(canvas, "S: 0~255", (420, 400),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv.putText(canvas, "V: 0~255", (420, 430),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv.imshow(win, canvas)

        key = cv.waitKey(16) & 0xFF
        if key == ord("q") or key == 27:
            break

    cv.destroyAllWindows()


if __name__ == "__main__":
    hsv_viewer()