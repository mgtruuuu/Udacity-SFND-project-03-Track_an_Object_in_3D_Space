#ifndef lidarData_hpp
#define lidarData_hpp

#include "dataStructures.h"

#include <fstream>
#include <string>





void cropLidarPoints(const float minX, const float maxX, const float maxY,
	const float minZ, const float maxZ, const float minR, std::vector<LidarPoint>& lidarPoints);

void loadLidarFromFile(const std::string filename, std::vector<LidarPoint>& lidarPoints);

void showLidarTopview(
	const std::vector<LidarPoint>& lidarPoints, 
	const cv::Size& worldSize, const cv::Size& imageSize, const bool bWait = true);

void showLidarImgOverlay(const cv::Mat& img, const std::vector<LidarPoint>& lidarPoints,
	const cv::Mat& P_rect_xx, const cv::Mat& R_rect_xx, const cv::Mat& RT,
	cv::Mat* extVisImg = nullptr, const bool bWait = true);


#endif /* lidarData_hpp */
