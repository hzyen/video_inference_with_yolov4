import cv2

img = cv2.imread('./data/image0001.jpg', cv2.IMREAD_UNCHANGED)

print('Original Dimensions : ',img.shape)

dim = (1276, 704)

resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

print('Resized Dimensions : ',resized_img.shape)

cv2.imwrite('./data/resized_image0001.jpg', resized_img)