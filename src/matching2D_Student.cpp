#include "matching2D.hpp"

#include <string>





void detectKeypoints(const Detector detectorType, const cv::Mat& imgGray,
    std::vector<cv::KeyPoint>& keypoints) {

    cv::Ptr<cv::FeatureDetector> detector;
    switch (detectorType) {
    case Detector::SHITOMASI:   detKeypointsShiTomasi(imgGray, keypoints);  break;

    case Detector::HARRIS:      detKeypointsHarris(imgGray, keypoints);     break;

    case Detector::FAST:        detKeypointsFAST(imgGray, keypoints);       break;


    case Detector::BRISK:       detector = cv::BRISK::create();
        detector->detect(imgGray, keypoints);       break;

    case Detector::ORB:         detector = cv::ORB::create();
        detector->detect(imgGray, keypoints);       break;

    case Detector::AKAZE:       detector = cv::AKAZE::create();
        detector->detect(imgGray, keypoints);       break;

    case Detector::SIFT:        detector = cv::SIFT::create();
        detector->detect(imgGray, keypoints);       break;

    //default:                    assert(false, "Wrong Detector type!\n");
    }
}


void clusterKPsWithROI(const std::vector<cv::KeyPoint>& keypoints, const float shrinkFactor,
    std::vector<BoundingBox>& boundingBoxes) {

    // Loop over all keypoints and associate them to a 2D bounding box.

    for (const auto& keypoint : keypoints) {

        std::vector<std::reference_wrapper<BoundingBox>> enclosingBoxes;

        for (auto& boundingBox : boundingBoxes) {

            // Shrink current bounding box slightly 
            // to avoid having too many outlier points around the edges.
            cv::Rect smallerBox;
            smallerBox.x = boundingBox.roi.x + static_cast<int>(boundingBox.roi.width * shrinkFactor / 2.0f);
            smallerBox.y = boundingBox.roi.y + static_cast<int>(boundingBox.roi.height * shrinkFactor / 2.0f);
            smallerBox.width = static_cast<int>(boundingBox.roi.width * (1 - shrinkFactor));
            smallerBox.height = static_cast<int>(boundingBox.roi.height * (1 - shrinkFactor));

            // Check whether point is within current bounding box.
            if (smallerBox.contains(keypoint.pt))
                enclosingBoxes.push_back(boundingBox);

        }   // eof loop over all bounding boxes


        //// Add Lidar point to bounding box.
        // BUT/// WHAT IF .size() >= 2 ???????????
        if (enclosingBoxes.size() >= 1)
            enclosingBoxes[0].get().keypoints.push_back(keypoint);
    }
}


void computeDescriptors(const Detector detectorType, const Descriptor descriptorType, const cv::Mat& imgGray,
    std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {

    cv::Ptr<cv::DescriptorExtractor> extractor;

    switch (descriptorType) {

    case Descriptor::BRIEF:     extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();    break;
    case Descriptor::FREAK:     extractor = cv::xfeatures2d::FREAK::create();                       break;

    case Descriptor::BRISK:     extractor = cv::BRISK::create();    break;
    case Descriptor::ORB:       extractor = cv::ORB::create();      break;
    case Descriptor::AKAZE:     extractor = cv::AKAZE::create();    break;
    case Descriptor::SIFT:      extractor = cv::SIFT::create();     break;

    //default:                    assert(false, "Wrong Descriptor type!\n");
    }

    //const cv::InputArray& mask{ cv::noArray() };
    //constexpr bool useProvidedKeypoints{ true };   // Detect keypoints and compute descriptor in two stages.
    //extractor->detectAndCompute(imgGray, mask, keypoints, descriptors, useProvidedKeypoints);
    extractor->compute(imgGray, keypoints, descriptors);
}







// Detect keypoints in image using the TRADITIONAL Shi-Thomasi detector.
void detKeypointsShiTomasi(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints) {

    // Shi-Tomasi detector

    constexpr int blockSize{ 4 };       // size of an average block for computing a derivative covariation matrix 
                                        // over each pixel neighborhood
    constexpr double maxOverlap{ 0.0 }; // maximun permissible overlap between two features in %
    constexpr double minDistance{ (1.0 - maxOverlap) * blockSize };
    const int maxCorners{ static_cast<int>(img.rows * img.cols / std::max(1.0, minDistance)) }; // max. num. of keypoints;

    constexpr double qualityLevel{ 0.01 };  // minimal accepted quality of image corners
    constexpr bool useHarrisDetector{ false };
    constexpr double k{ 0.04 };


    // Apply corner detection.
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance,
        cv::Mat{}, blockSize, useHarrisDetector, k);


    // Add corners to result vector.
    for (const auto& corner : corners) {
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f{ corner.x, corner.y };
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
}


void detKeypointsHarris(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints) {

    // detector parameters

    constexpr int blockSize{ 2 };       // For every pixel, a blockSize × blockSize neighborhood is considered.
    constexpr int apertureSize{ 3 };    // aperture parameter for Sobel operator (must be odd)
    constexpr int minResponse{ 100 };   // minimum value for a corner in the 8bit scaled response matrix
    constexpr double k{ 0.04 };         // Harris parameter (see equation for details)


    // Detect Harris corners and normalize output

    cv::Mat dst{ cv::Mat::zeros(img.size(), CV_32FC1) };
    cv::Mat dst_norm;
    cv::Mat dst_norm_scaled;

    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    normalize(dst, dst_norm, .0, 255.0, cv::NORM_MINMAX, CV_32FC1, cv::Mat{});
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);


    // Apply non-maximum suppression (NMS)

    constexpr float maxOverlap{ 0.0f };     // maximum permissible overlap between two features in %, 
                                            // used during non-maxima suppression
    for (int j{ 0 }; j < dst_norm.rows; ++j) {
        for (int i{ 0 }; i < dst_norm.cols; ++i) {
            const float response{ dst_norm.at<float>(j, i) };

            // Apply the minimum threshold for Harris cornerness response.

            if (response < minResponse)     continue;

            // Create a tentative new keypoint otherwise.
            cv::KeyPoint newKeyPoint;
            newKeyPoint.pt = static_cast<cv::Point2f>(cv::Point2i{ i, j });
            newKeyPoint.size = static_cast<float>(2 * apertureSize);
            newKeyPoint.response = response;


            // Perform non-maximum suppression (NMS) in local neighbourhood around the new keypoint.

            bool bOverlap{ false };

            // Loop over all existing keypoints.
            for (auto& keypoint : keypoints) {
                const float kptOverlap{ cv::KeyPoint::overlap(newKeyPoint, keypoint) };

                // Test if overlap exceeds the maximum percentage allowable.
                if (kptOverlap > maxOverlap) {
                    bOverlap = true;

                    // If overlapping, test if new response is the local maximum.
                    if (newKeyPoint.response > keypoint.response) {
                        keypoint = newKeyPoint;     // Replace the old keypoint.
                        break;                      // Exit for loop.
                    }
                }
            }

            // If above response threshold and not overlapping any other keypoint,
            // add to keypoints list.
            if (!bOverlap)  keypoints.push_back(newKeyPoint);
        }
    }
}


void detKeypointsFAST(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints) {

    constexpr int threshold{ 100 };
    constexpr bool nonmaxSuppresion{ true };
    constexpr cv::FastFeatureDetector::DetectorType type{ cv::FastFeatureDetector::TYPE_9_16 };

    cv::FAST(img, keypoints, threshold, nonmaxSuppresion, type);
}




// Find best matches for keypoints in two camera images based on several matching methods.
void matchDescriptors(const cv::Mat& descSource, const cv::Mat& descRef,
    const Matcher matcherType, const DescriptorOption descriptorOptionType,
    const Selector selectorType, const bool crossCheck, std::vector<cv::DMatch>& matches) {

    cv::Ptr<cv::DescriptorMatcher> matcher;     // configure matcher

    if (matcherType == Matcher::MAT_BF) {

        // for BRISK, BRIEF, ORB, FREAK and AKAZE descriptors
        if (descriptorOptionType == DescriptorOption::DES_BINARY) {
            const int normType{ cv::NORM_HAMMING };
            matcher = cv::BFMatcher::create(normType, crossCheck);
        }

        // for SIFT descriptor
        else if (descriptorOptionType == DescriptorOption::DES_HOG) {
            const int normType{ cv::NORM_L2 };
            matcher = cv::BFMatcher::create(normType, crossCheck);
        }
    }
    else if (matcherType == Matcher::MAT_FLANN) {
        matcher = cv::FlannBasedMatcher::create();
    }




    // Perform matching task.

    if (selectorType == Selector::SEL_NN) {                 // nearest neighbor (best match)
        matcher->match(descSource, descRef, matches);       // Finds the best match for each descriptor in desc1.
    }
    else if (selectorType == Selector::SEL_KNN) {           // k nearest neighbors (k=2)
        //assert(crossCheck == false, "The 8th argument of the function matchDescriptors() in main() must be 'false' in order to choose the SEL_KNN Selector Type.\n");
        constexpr int k{ 2 };
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, k);

        constexpr float minDescDistRatio{ 0.8f };
        for (const auto& knn_match : knn_matches)
            if (knn_match[0].distance < minDescDistRatio * knn_match[1].distance)
                matches.push_back(knn_match[0]);
        std::cout << '\t' << knn_matches.size() - matches.size() << " keypoints were removed (K-Nearest-Neighbor approach)." << std::endl;
    }
}




void visualizeKeypoints(const cv::Mat& img, const std::vector<cv::KeyPoint>& keypoints, const bool bVis) {

    if (bVis == false)      return;


    cv::Mat matchImg{ img.clone() };
    const cv::Rect& vehicleRect{ 535, 180, 180, 150 };
    cv::rectangle(matchImg, vehicleRect, cv::Scalar{ 255.0, 255.0, 255.0 }, 1, 20, 0);
    cv::Mat img_kps;
    cv::drawKeypoints(matchImg, keypoints, img_kps);

    std::string windowName{ "Matching keypoints between two camera images" };
    cv::namedWindow(windowName, 7);
    cv::imshow(windowName, img_kps);
    cv::waitKey(0);
}

void visualizeMatches(const cv::Mat& imgFront, const cv::Mat& imgBack,
    const std::vector<cv::KeyPoint>& keypointsFront, const std::vector<cv::KeyPoint>& keypointsBack,
    const std::vector<cv::DMatch>& matches, const bool bVis) {

    if (bVis == false)      return;


    cv::Mat matchImg{ imgBack.clone() };
    cv::drawMatches(
        imgFront, keypointsFront, imgBack, keypointsBack, matches, matchImg,
        cv::Scalar::all(-1), cv::Scalar::all(-1),
        std::vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    const std::string windowName{ "Matching keypoints between two camera images" };
    cv::namedWindow(windowName, 7);
    cv::imshow(windowName, matchImg);
    std::cout << "Press key to continue to next image" << std::endl;
    cv::waitKey(0);     // wait for key to be pressed
}