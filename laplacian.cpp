#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <cmath>

static inline int reflect101(int p, int len) {
    // BORDER_REFLECT_101: -1->1, -2->2, len->len-2, len+1->len-3 ...
    if (len <= 1) return 0;
    while (p < 0 || p >= len) {
        if (p < 0) p = -p;                 // -1 -> 1
        else       p = 2 * len - 2 - p;    // len -> len-2
    }
    return p;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Uzycie: " << argv[0] << " <obraz>\n";
        return 1;
    }

    cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Nie mozna wczytac obrazu: " << argv[1] << "\n";
        return 1;
    }

    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // ============================================================
    //  A) OpenCV Laplacian (zostaje) -> laplacian.png
    // ============================================================
    cv::Mat lap16;
    int ksize = 3;
    cv::Laplacian(gray, lap16, CV_16S, ksize, 1.0, 0.0, cv::BORDER_DEFAULT);

    cv::Mat lap8;
    cv::convertScaleAbs(lap16, lap8);
    cv::imwrite("laplacian.png", lap8);
    std::cout << "Zapisano: laplacian.png\n";

    // ============================================================
    //  B) Manual: Laplacian jak w OpenCV dla ksize=3:
    //     lap = d2x(Sobel) + d2y(Sobel) + BORDER_DEFAULT (reflect101)
    //     -> laplacian_i.png
    // ============================================================
    cv::Mat lap16_i(gray.rows, gray.cols, CV_16S, cv::Scalar(0));

    auto I = [&](int y, int x) -> int {
        y = reflect101(y, gray.rows);
        x = reflect101(x, gray.cols);
        return (int)gray.at<uchar>(y, x);
    };

    for (int y = 0; y < gray.rows; ++y) {
        for (int x = 0; x < gray.cols; ++x) {
            // d2x: ( [1 -2 1] w X ) + wygładzanie [1 2 1] w Y
            int d2x =
                1 * ( I(y-1, x-1) - 2*I(y-1, x) + I(y-1, x+1) ) +
                2 * ( I(y  , x-1) - 2*I(y  , x) + I(y  , x+1) ) +
                1 * ( I(y+1, x-1) - 2*I(y+1, x) + I(y+1, x+1) );

            // d2y: ( [1 -2 1] w Y ) + wygładzanie [1 2 1] w X
            int d2y =
                1 * ( I(y-1, x-1) - 2*I(y, x-1) + I(y+1, x-1) ) +
                2 * ( I(y-1, x  ) - 2*I(y, x  ) + I(y+1, x  ) ) +
                1 * ( I(y-1, x+1) - 2*I(y, x+1) + I(y+1, x+1) );

            int v = d2x + d2y;

            v = std::max(-32768, std::min(32767, v));
            lap16_i.at<short>(y, x) = (short)v;
        }
    }

    // jak convertScaleAbs: abs + saturacja do 8-bit
    cv::Mat lap8_i(gray.rows, gray.cols, CV_8U);
    for (int y = 0; y < lap16_i.rows; ++y) {
        for (int x = 0; x < lap16_i.cols; ++x) {
            int v = std::abs((int)lap16_i.at<short>(y, x));
            if (v > 255) v = 255;
            lap8_i.at<uchar>(y, x) = (uchar)v;
        }
    }

    cv::imwrite("laplacian_i.png", lap8_i);
    std::cout << "Zapisano: laplacian_i.png\n";

    return 0;
}