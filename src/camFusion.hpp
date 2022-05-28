#ifndef camFusion_hpp
#define camFusion_hpp

#include "dataStructures.h"

#include <opencv2/core.hpp>

#include <vector>





void clusterLidarWithROI(const std::vector<LidarPoint>& lidarPoints,
    const cv::Mat& P_rect_xx, const cv::Mat& R_rect_xx, const cv::Mat& RT, const float shrinkFactor,
    std::vector<BoundingBox>& boundingBoxes);


void show3DObjects(const std::vector<BoundingBox>& boundingBoxes,
    const cv::Size& worldSize, const cv::Size& imageSize);



void matchBoundingBoxes(const std::vector<cv::DMatch>& curr_KPMatches,
    const std::vector<cv::KeyPoint>& prev_KPs, const std::vector<cv::KeyPoint>& curr_KPs,
    const std::vector<BoundingBox>& prev_BBoxes, const std::vector<BoundingBox>& curr_BBoxes,
    std::map<int, int>& bbBestMatches);


std::pair<double, double> computeTTCLidar(const std::vector<LidarPoint>& lidarPointsPrev,
    const std::vector<LidarPoint>& lidarPointsCurr, const double frameRate, double& TTC);

void clusterKptMatchesWithROI(
    const std::vector<cv::KeyPoint>& kptsPrev, const std::vector<cv::KeyPoint>& kptsCurr,
    const std::vector<cv::DMatch>& kptMatches, BoundingBox* boundingBox);

void computeTTCCamera(
    const std::vector<cv::KeyPoint>& kptsPrev, const std::vector<cv::KeyPoint>& kptsCurr,
    const std::vector<cv::DMatch>& kptMatches, const double frameRate, double& TTC);






#endif /* camFusion_hpp */