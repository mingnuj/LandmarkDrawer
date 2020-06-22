import numpy as np
import cv2
import collections
import os

# image path / landmark path
image_path = "./landmark/Test_image/"
result_path = "./landmark/Test_result/result_numpy_68_2/"

results = []
images = []

# landmark and image file lists
for x in os.listdir(result_path):
    results.append(os.path.join(result_path, x))
    images.append(os.path.join(image_path, "{}jpg".format(x[:-3])))

# landmark format
# list 0 - 16: face, 17 - 21: eyebrow, ..., 60 - 67: teeth
pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
              'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
              'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
              'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
              'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
              'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
              'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
              'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
              'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
              }

for num in range(len(results)):
    # landmark file reading, image file reading
    preds = np.load(results[num])
    image = cv2.imread(images[num])

    # drawing circles to image
    for pred_type in pred_types.values():
        for idx, point in enumerate(np.column_stack((preds[pred_type.slice, 0].T, preds[pred_type.slice, 1].T))):
            colors = [i * 255 for i in list(pred_type.color)]
            image = cv2.circle(image, (point[0], point[1]), 1, color=colors, thickness=-1)

    cv2.imshow("plot", image)
    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break