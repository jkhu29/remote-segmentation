import os
import cv2
import numpy as np
import tfrecord

cnt = 0
writer = tfrecord.TFRecordWriter("test.tfrecord")
for image_name in os.listdir("test_images/images"):
    cnt += 1
    image = cv2.imread(os.path.join("test_images/images", image_name))
    image = image.transpose(2, 0, 1)
    writer.write({
        "image": (image.tobytes(), "byte"),
        "size": (512, "int"),
    })

writer.close()
