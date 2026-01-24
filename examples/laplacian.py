import cv2
import torch
import fastcv

img = cv2.imread("../artifacts/grayscale.jpg", cv2.IMREAD_GRAYSCALE)
img_tensor = torch.from_numpy(img).cuda()
laplacian_tensor = fastcv.laplacian(img_tensor)
laplacian_np = laplacian_tensor.cpu().numpy()
cv2.imwrite("output_laplacian.jpg", laplacian_np)

print("Saved Laplacian image.")