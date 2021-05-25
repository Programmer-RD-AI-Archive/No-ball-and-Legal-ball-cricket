import cv2

config = {
    "testing": {
        "img_size": 112,
        "NUM_CLASS": 2,
        'color_channel(s)':3,
        "convtofc": 128 * 100 * 100,
        "imread_type": cv2.IMREAD_COLOR,
    }
}
