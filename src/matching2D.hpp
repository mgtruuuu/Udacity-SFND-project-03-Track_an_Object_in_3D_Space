#ifndef matching2D_hpp
#define matching2D_hpp

#include "dataStructures.h"
#include "options.h"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>










void detectKeypoints(const Detector detectorType, const cv::Mat& imgGray,
    std::vector<cv::KeyPoint>& keypoints);

void clusterKPsWithROI(const std::vector<cv::KeyPoint>& keypoints, const float shrinkFactor,
    std::vector<BoundingBox>& boundingBoxes);

void computeDescriptors(const Detector detectorType, const Descriptor descriptorType, const cv::Mat& imgGray,
    std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

void detKeypointsShiTomasi(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints);

void detKeypointsHarris(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints);

void detKeypointsFAST(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints);

void matchDescriptors(const cv::Mat& descSource, const cv::Mat& descRef,
    const Matcher matcherType, const DescriptorOption descriptorOptionType,
    const Selector selectorType, const bool crossCheck, std::vector<cv::DMatch>& matches);

void visualizeKeypoints(const cv::Mat& img, const std::vector<cv::KeyPoint>& keypoints, const bool bVis);

void visualizeMatches(const cv::Mat& imgFront, const cv::Mat& imgBack,
    const std::vector<cv::KeyPoint>& keypointsFront, const std::vector<cv::KeyPoint>& keypointsBack,
    const std::vector<cv::DMatch>& matches, const bool bVis);





#endif /* matching2D_hpp */
