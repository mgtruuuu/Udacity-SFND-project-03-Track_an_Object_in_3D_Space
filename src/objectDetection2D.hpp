#ifndef objectDetection2D_hpp
#define objectDetection2D_hpp

#include <stdio.h>
#include <opencv2/core.hpp>

#include "dataStructures.h"

void detectObjects(const cv::Mat& img, const float confThreshold, const float nmsThreshold,
    const std::string classesFile, const std::string modelConfiguration,
    const std::string modelWeights, const bool bVis, std::vector<BoundingBox>& bBoxes);


#endif /* objectDetection2D_hpp */
