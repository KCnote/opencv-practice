import cv2 as cv
import numpy as np


class PixelViewer:
    def __init__(self, image_path):
        self.img = cv.imread(image_path)
        if self.img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        self.img_h, self.img_w = self.img.shape[:2]

        self.win_name = "Pixel Viewer"
        self.win_w = 1200
        self.win_h = 800

        self.zoom = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 80.0

        self.center_x = self.img_w / 2
        self.center_y = self.img_h / 2

        self.is_dragging = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.mouse_x = 0
        self.mouse_y = 0

        self.grid_zoom_threshold = 12.0
        self.pixel_value_zoom_threshold = 35.0
        self.max_text_pixels = 600

        cv.namedWindow(self.win_name, cv.WINDOW_NORMAL)
        cv.resizeWindow(self.win_name, self.win_w, self.win_h)
        cv.setMouseCallback(self.win_name, self.on_mouse)

    def put_pixel_text(self, img, text, org, scale=0.32):
        # darker gold outline, white text
        cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv.LINE_AA)

    def screen_to_image(self, sx, sy):
        img_x = self.center_x + (sx - self.win_w / 2) / self.zoom
        img_y = self.center_y + (sy - self.win_h / 2) / self.zoom
        return img_x, img_y

    def on_mouse(self, event, x, y, flags, param):
        self.mouse_x = x
        self.mouse_y = y

        if event == cv.EVENT_LBUTTONDOWN:
            self.is_dragging = True
            self.last_mouse_x = x
            self.last_mouse_y = y

        elif event == cv.EVENT_LBUTTONUP:
            self.is_dragging = False

        elif event == cv.EVENT_MOUSEMOVE and self.is_dragging:
            dx = x - self.last_mouse_x
            dy = y - self.last_mouse_y

            self.center_x -= dx / self.zoom
            self.center_y -= dy / self.zoom
            self.clamp_center()

            self.last_mouse_x = x
            self.last_mouse_y = y

        elif event == cv.EVENT_MOUSEWHEEL:
            before_x, before_y = self.screen_to_image(x, y)

            self.zoom *= 1.25 if flags > 0 else 1 / 1.25
            self.zoom = max(self.min_zoom, min(self.zoom, self.max_zoom))

            after_x, after_y = self.screen_to_image(x, y)

            self.center_x += before_x - after_x
            self.center_y += before_y - after_y
            self.clamp_center()

    def clamp_center(self):
        self.center_x = max(0, min(self.center_x, self.img_w - 1))
        self.center_y = max(0, min(self.center_y, self.img_h - 1))

    def get_visible_region(self):
        visible_w = self.win_w / self.zoom
        visible_h = self.win_h / self.zoom

        x0 = int(np.floor(self.center_x - visible_w / 2))
        y0 = int(np.floor(self.center_y - visible_h / 2))
        x1 = int(np.ceil(self.center_x + visible_w / 2))
        y1 = int(np.ceil(self.center_y + visible_h / 2))

        src_x0 = max(0, x0)
        src_y0 = max(0, y0)
        src_x1 = min(self.img_w, x1)
        src_y1 = min(self.img_h, y1)

        return x0, y0, src_x0, src_y0, src_x1, src_y1

    def draw_grid_and_values(self, canvas, x0, y0, src_x0, src_y0, src_x1, src_y1):
        if self.zoom < self.grid_zoom_threshold:
            return

        # grid only
        for ix in range(src_x0, src_x1 + 1):
            sx = int((ix - x0) * self.zoom)
            if 0 <= sx < self.win_w:
                cv.line(canvas, (sx, 0), (sx, self.win_h), (80, 80, 80), 1)

        for iy in range(src_y0, src_y1 + 1):
            sy = int((iy - y0) * self.zoom)
            if 0 <= sy < self.win_h:
                cv.line(canvas, (0, sy), (self.win_w, sy), (80, 80, 80), 1)

        if self.zoom < self.pixel_value_zoom_threshold:
            return

        visible_pixel_count = (src_x1 - src_x0) * (src_y1 - src_y0)
        if visible_pixel_count > self.max_text_pixels:
            return

        for iy in range(src_y0, src_y1):
            sy = int((iy - y0) * self.zoom)

            if sy < 0 or sy >= self.win_h:
                continue

            for ix in range(src_x0, src_x1):
                sx = int((ix - x0) * self.zoom)

                if sx < 0 or sx >= self.win_w:
                    continue

                b, g, r = self.img[iy, ix]

                values = [str(b), str(g), str(r)]

                font = cv.FONT_HERSHEY_SIMPLEX
                scale = 0.32
                thickness = 1

                line_gap = int(self.zoom * 0.22)

                total_height = line_gap * (len(values) - 1)

                start_y = sy + int((self.zoom - total_height) / 2)

                for i, value in enumerate(values):
                    text_size, _ = cv.getTextSize(
                        value,
                        font,
                        scale,
                        thickness
                    )

                    text_w = text_size[0]
                    text_h = text_size[1]

                    tx = sx + int((self.zoom - text_w) / 2)
                    ty = start_y + i * line_gap + int(text_h / 2)

                    self.put_pixel_text(
                        canvas,
                        value,
                        (tx, ty),
                        scale
                    )
                    
    def draw_top_bar(self, canvas):
        text = "drag: pan | wheel: zoom | r: reset | q/esc: quit"

        cv.rectangle(canvas, (0, 0), (self.win_w, 32), (30, 30, 30), -1)
        cv.putText(canvas, text, (10, 23), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv.LINE_AA)

    def draw_status_bar(self, canvas):
        mouse_img_x, mouse_img_y = self.screen_to_image(self.mouse_x, self.mouse_y)

        if 0 <= mouse_img_x < self.img_w and 0 <= mouse_img_y < self.img_h:
            ix = int(mouse_img_x)
            iy = int(mouse_img_y)
            b, g, r = self.img[iy, ix]
            text = f"x={ix}, y={iy} | BGR=({b},{g},{r}) | zoom={self.zoom:.2f}x"
        else:
            text = f"outside image | zoom={self.zoom:.2f}x"

        bar_h = 34
        cv.rectangle(canvas, (0, self.win_h - bar_h), (self.win_w, self.win_h), (30, 30, 30), -1)

        text_size, _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        x = max(10, self.win_w - text_size[0] - 15)
        y = self.win_h - 10

        # 상태표시줄은 그냥 흰색
        cv.putText(canvas, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv.LINE_AA)

    def draw(self):
        canvas = np.full((self.win_h, self.win_w, 3), 40, dtype=np.uint8)

        x0, y0, src_x0, src_y0, src_x1, src_y1 = self.get_visible_region()

        if src_x0 < src_x1 and src_y0 < src_y1:
            patch = self.img[src_y0:src_y1, src_x0:src_x1]

            dst_x0 = int((src_x0 - x0) * self.zoom)
            dst_y0 = int((src_y0 - y0) * self.zoom)
            dst_w = max(1, int(patch.shape[1] * self.zoom))
            dst_h = max(1, int(patch.shape[0] * self.zoom))

            interp = cv.INTER_NEAREST if self.zoom >= 1 else cv.INTER_AREA
            resized = cv.resize(patch, (dst_w, dst_h), interpolation=interp)

            dst_x1 = dst_x0 + dst_w
            dst_y1 = dst_y0 + dst_h

            cx0 = max(0, dst_x0)
            cy0 = max(0, dst_y0)
            cx1 = min(self.win_w, dst_x1)
            cy1 = min(self.win_h, dst_y1)

            if cx1 > cx0 and cy1 > cy0:
                rx0 = cx0 - dst_x0
                ry0 = cy0 - dst_y0
                rx1 = rx0 + (cx1 - cx0)
                ry1 = ry0 + (cy1 - cy0)

                canvas[cy0:cy1, cx0:cx1] = resized[ry0:ry1, rx0:rx1]

            self.draw_grid_and_values(canvas, x0, y0, src_x0, src_y0, src_x1, src_y1)

        self.draw_top_bar(canvas)
        self.draw_status_bar(canvas)

        cv.imshow(self.win_name, canvas)

    def run(self):
        while True:
            self.draw()
            key = cv.waitKey(16) & 0xFF

            if key == ord("q") or key == 27:
                break

            if key == ord("r"):
                self.zoom = 1.0
                self.center_x = self.img_w / 2
                self.center_y = self.img_h / 2

        cv.destroyAllWindows()


if __name__ == "__main__":
    viewer = PixelViewer("data/cat.png")
    viewer.run()