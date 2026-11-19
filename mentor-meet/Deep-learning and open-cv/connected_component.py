import cv2
import numpy as np

image = cv2.imread("./Photos/german-shepd.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(
    gray, 127, 255, cv2.THRESH_BINARY_INV
)

num_labels, labels, stats, centroids = (
    cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )
)

output = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

for label in range(1, num_labels):
    x = stats[label, cv2.CC_STAT_LEFT]
    y = stats[label, cv2.CC_STAT_TOP]
    width = stats[label, cv2.CC_STAT_WIDTH]
    height = stats[label, cv2.CC_STAT_HEIGHT]
    area = stats[label, cv2.CC_STAT_AREA]

    # Ignore tiny objects (noise)
    if area < 100:
        continue

    # Draw bounding box
    cv2.rectangle(
        output,
        (x, y),
        (x + width, y + height),
        (0, 255, 0),
        2
    )

    print(f"Object {label}: Area = {area}")
    print(f"Position: ({x}, {y})")
    print(f"Size: {width} x {height}")
    print(f"Centroid: {centroids[label]}")

print("Total labels including background:", num_labels)
print("Objects before filtering:", num_labels - 1)

cv2.imshow("Binary Image", binary)
cv2.imshow("CCA Result", output)

cv2.waitKey(0)
cv2.destroyAllWindows()
