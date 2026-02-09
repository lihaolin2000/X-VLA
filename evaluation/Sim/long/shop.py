import cv2
with open("text.txt", "w") as f:
    f.write(cv2.getBuildInformation())

import numpy as np
# print(cv2.getBuildInformation())
h, w = 256, 256
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('test.avi', fourcc, 30, (w, h))

# fourcc = cv2.VideoWriter_fourcc(*'avc1')

# out = cv2.VideoWriter('test.mp4', fourcc, 30, (w, h))
print("opened:", out.isOpened())

for i in range(60):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(frame, f'{i}', (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
    out.write(frame)

out.release()