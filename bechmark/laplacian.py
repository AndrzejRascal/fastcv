import time
import cv2
import torch
import fastcv
import numpy as np


def benchmark_laplacian(sizes=[1024, 2048, 4096], runs=50):
    results = []

    for size in sizes:
        print(f"\n=== Benchmarking Laplacian {size}x{size} image ===")

        img_np = np.random.randint(0, 256, (size, size), dtype=np.uint8)
        img_torch = torch.from_numpy(img_np).cuda()

        start = time.perf_counter()
        for _ in range(runs):
            _ = cv2.Laplacian(img_np, cv2.CV_8U, ksize=3)

        end = time.perf_counter()
        cv_time = (end - start) / runs * 1000

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            _ = fastcv.laplacian(img_torch)
        torch.cuda.synchronize()
        end = time.perf_counter()
        fc_time = (end - start) / runs * 1000

        results.append((size, cv_time, fc_time))
        print(f"OpenCV (CPU): {cv_time:.4f} ms | fastcv (CUDA): {fc_time:.4f} ms")

    return results


if __name__ == "__main__":
    results = benchmark_laplacian()
    print("\n=== Final Results: Laplacian ===")
    print("Size\t\tOpenCV (CPU)\tfastcv (CUDA)")
    for size, cv_time, fc_time in results:
        print(f"{size}x{size}\t{cv_time:.4f} ms\t{fc_time:.4f} ms")