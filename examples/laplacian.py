import cv2
import torch
import fastcv

img = cv2.imread("../artifacts/test.jpg", cv2.IMREAD_GRAYSCALE)
img_tensor = torch.from_numpy(img).cuda()
laplacian_tensor = fastcv.laplacian(img_tensor)
laplacian_np = laplacian_tensor.cpu().numpy()
cv2.imwrite("output_laplacian.jpg", laplacian_np)

laplacian_opencv = cv2.Laplacian(img, cv2.CV_8U, ksize=1)
cv2.imwrite("output_laplacian_opencv.jpg", laplacian_opencv)

laplacian_compare_tensor = fastcv.laplacian_compare(img_tensor)
laplacian_compare_np = laplacian_compare_tensor.cpu().numpy()
cv2.imwrite("output_laplacian_compare.jpg", laplacian_compare_np)

print("Saved Laplacian image.")