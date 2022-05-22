#include <algorithm>
#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "lidarData.hpp"
#include "enums.h"



// Remove Lidar points based on min. and max distance in X, Y and Z.
void cropLidarPoints(const float minX, const float maxX, const float maxY,
    const float minZ, const float maxZ, const float minR, std::vector<LidarPoint>& lidarPoints) {
    
    std::vector<LidarPoint> newLidarPts;

    for (const auto& lidarPoint : lidarPoints) {

        // Check if Lidar point is outside of boundaries.
        if (lidarPoint.x >= minX && lidarPoint.x <= maxX
            && lidarPoint.z >= minZ && lidarPoint.z <= maxZ && lidarPoint.z <= 0.0
            && abs(lidarPoint.y) <= maxY && lidarPoint.r >= minR) {
            newLidarPts.push_back(lidarPoint);
        }
    }

    lidarPoints = newLidarPts;
}



// Load Lidar points from a given location and store them in a vector.
void loadLidarFromFile(const std::string filename, std::vector<LidarPoint>& lidarPoints) {

    // Allocate 4 MB buffer (only ~130*4*4 KB are needed).
    long num{ 1'000'000 };
    float* data = new float[num];

    // pointers
    float* px{ data + 0 };
    float* py{ data + 1 };
    float* pz{ data + 2 };
    float* pr{ data + 3 };

    // Load point cloud.
    FILE* stream{ fopen(filename.c_str(), "rb") };
    num = fread(data, sizeof(float), num, stream) / 4;

    

    for (int32_t i{ 0 }; i < num; ++i) {
        LidarPoint lpt{ 
            static_cast<double>(*px), static_cast<double>(*py), 
            static_cast<double>(*pz), static_cast<double>(*pr) 
        };
        lidarPoints.push_back(lpt);
        px += 4; py += 4; pz += 4; pr += 4;
    }
    
    fclose(stream);

    delete[] data;
}


void showLidarTopview(const std::vector<LidarPoint>& lidarPoints, 
    const cv::Size& worldSize, const cv::Size& imageSize, const bool bWait) {

    // Create topview image.
    cv::Mat topviewImg{ imageSize, CV_8UC3, cv::Scalar{ .0, .0, .0 } };

    // Plot Lidar points into image
    for (const auto& lidarPoint : lidarPoints) {
        float xw{ static_cast<float>(lidarPoint.x) };       // world position in m with x facing forward from sensor
        float yw{ static_cast<float>(lidarPoint.y) };       // world position in m with y facing left from sensor

        int y{ static_cast<int>((-xw * imageSize.height / worldSize.height)) + imageSize.height };
        int x{ static_cast<int>((-yw * imageSize.width / worldSize.width)) + imageSize.width / 2 };

        cv::circle(topviewImg, cv::Point{ x, y }, 5, cv::Scalar(.0, .0, 255.0), -1);
    }

    // Plot distance markers.
    float lineSpacing{ 2.0f };          // gap between distance markers
    int nMarkers{ static_cast<int>(floor(worldSize.height / lineSpacing)) };
    for (int i{ 0 }; i < nMarkers; ++i) {
        int y{ static_cast<int>((-(i * lineSpacing) * imageSize.height / worldSize.height)) + imageSize.height };
        cv::line(topviewImg, cv::Point{ 0, y }, cv::Point{ imageSize.width, y }, cv::Scalar{ 255.0, .0, .0 });
    }

    // Display image.
    std::string windowName{ "Top-View Perspective of LiDAR data" };
    cv::namedWindow(windowName, 2);
    cv::imshow(windowName, topviewImg);
    if (bWait)      cv::waitKey(0);     // Wait for key to be pressed.
}


void showLidarImgOverlay(const cv::Mat& img, const std::vector<LidarPoint>& lidarPoints,
    const cv::Mat& P_rect_xx, const cv::Mat& R_rect_xx, const cv::Mat& RT,
    cv::Mat* extVisImg, const bool bWait) {

    // Init image for visualization.
    cv::Mat visImg;
    if (extVisImg == nullptr)   
        visImg = img.clone();
    else
        visImg = *extVisImg;

    cv::Mat overlay{ visImg.clone() };

    // Find max. x-value.
    double maxVal{ 0.0 };
    for (auto& lidarPoint : lidarPoints)
        maxVal = std::max(lidarPoint.x, maxVal);

    cv::Mat X{ 4, 1, cv::DataType<double>::type };
    cv::Mat Y{ 3, 1, cv::DataType<double>::type };

    for (const auto& lidarPoint : lidarPoints) {

        // Assemble vector for matrix-vector-multiplication.
        X.at<double>(0, 0) = lidarPoint.x;
        X.at<double>(1, 0) = lidarPoint.y;
        X.at<double>(2, 0) = lidarPoint.z;
        X.at<double>(3, 0) = 1;

        // Project Lidar point into camera.
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);

        double val{ lidarPoint.x };
        double red{ std::min(255.0, 255.0 * abs((val - maxVal) / maxVal)) };
        double green{ std::min(255.0, 255.0 * (1 - abs((val - maxVal) / maxVal))) };
        cv::circle(overlay, pt, 5, cv::Scalar{ .0, green, red }, -1);
    }

    double opacity{ 0.6 };
    cv::addWeighted(overlay, opacity, visImg, 1 - opacity, .0, visImg);

    // Return augmented image or wait if no image has been provided.
    if (extVisImg == nullptr) {
        std::string windowName{ "LIDAR data on image overlay" };
        cv::namedWindow(windowName, 3);
        cv::imshow(windowName, visImg);
        if (bWait)      cv::waitKey(0);     // Wait for key to be pressed
    }
    else {
        extVisImg = &visImg;
    }
}

