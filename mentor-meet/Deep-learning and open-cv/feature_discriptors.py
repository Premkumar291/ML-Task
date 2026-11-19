import cv2

image = cv2.imread("./Photos/german-shepd.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(nfeatures=500)

keypoints, descriptors = orb.detectAndCompute(gray, None)

output = cv2.drawKeypoints(
    image,
    keypoints,
    None,
    color=(0, 255, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)


print("Number of keypoints:", len(keypoints))
print("Descriptor shape:", descriptors.shape if descriptors is not None else None)

cv2.imshow("Original Image", image)
cv2.imshow("ORB Features", output)

cv2.waitKey(0)
cv2.destroyAllWindows()