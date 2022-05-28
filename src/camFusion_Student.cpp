#include "camFusion.hpp"
#include "dataStructures.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <deque>
#include <iostream>
#include <set>




// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(const std::vector<LidarPoint>& lidarPoints, 
    const cv::Mat& P_rect_xx, const cv::Mat& R_rect_xx, const cv::Mat& RT, const float shrinkFactor,
    std::vector<BoundingBox>& boundingBoxes) {
    
    // Loop over all Lidar points and associate them to a 2D bounding box.
    cv::Mat X{ 4, 1, cv::DataType<double>::type };
    cv::Mat Y{ 3, 1, cv::DataType<double>::type };

    for (const auto& lidarPoint : lidarPoints) {

        // Assemble vector for matrix-vector-multiplication.
        X.at<double>(0, 0) = lidarPoint.x;
        X.at<double>(1, 0) = lidarPoint.y;
        X.at<double>(2, 0) = lidarPoint.z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = static_cast<int>(Y.at<double>(0, 0) / Y.at<double>(2, 0));
        pt.y = static_cast<int>(Y.at<double>(1, 0) / Y.at<double>(2, 0));


        //std::vector<std::reference_wrapper<BoundingBox*>> enclosingBoxes;
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
            if (smallerBox.contains(pt))
                enclosingBoxes.push_back(boundingBox);

        }   // eof loop over all bounding boxes

        // Add Lidar point to bounding box.
        if (enclosingBoxes.size() == 1)
            enclosingBoxes[0].get().lidarPoints.push_back(lidarPoint);

    }   // eof loop over all Lidar points
}




/*
* The show3DObjects() function below can handle different output image sizes, 
* but the text output has been manually tuned to fit the 2000x2000 size.
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/

void show3DObjects(const std::vector<BoundingBox>& boundingBoxes, 
    const cv::Size& worldSize, const cv::Size& imageSize) {
    
    // Create topview image.
    cv::Mat topviewImg{ imageSize, CV_8UC3, cv::Scalar{ 255.0, 255.0, 255.0 } };

    for (const auto& boundingBox : boundingBoxes) {
        cv::RNG rng{ static_cast<uint64>(boundingBox.boxID) };
        cv::Scalar currColor{ static_cast<double>(rng.uniform(0, 150)),
            static_cast<double>(rng.uniform(0, 150)),
            static_cast<double>(rng.uniform(0, 150)) };

        // Plot Lidar points into top view image.
        int top{ INT_MAX };
        int left{ INT_MAX };
        int bottom{ 0 };
        int right{ 0 };

        float xwmin{ FLT_MAX };
        float ywmin{ FLT_MAX };
        float ywmax{ FLT_MIN };

        for (const auto& lidarPoint : boundingBox.lidarPoints) {
            float xw{ static_cast<float>(lidarPoint.x) };       // world position in m with x facing forward from sensor
            float yw{ static_cast<float>(lidarPoint.y) };       // world position in m with y facing left from sensor
            xwmin = std::min(xwmin, xw);
            ywmin = std::min(ywmin, yw);
            ywmax = std::max(ywmax, yw);

            // top-view coordinates
            // 1. width/height proportion
            // 2. point(origin) symmetry (Note y-axis of Lidarpoint coordinate)
            // 3. translation ( +1/2*(imageSize.width), +(imageSize.height) )
            int x{ static_cast<int>((-yw * imageSize.width / worldSize.width)) + imageSize.width / 2 };
            int y{ static_cast<int>((-xw * imageSize.height / worldSize.height)) + imageSize.height };

            // Find enclosing rectangle.
            top = std::min(top, y);
            left = std::min(left, x);
            bottom = std::max(bottom, y);
            right = std::max(right, x);

            // Draw individual point.
            cv::circle(topviewImg, cv::Point{ x, y }, 4, currColor, -1);
        }

        // Draw enclosing rectangle.
        cv::rectangle(topviewImg, cv::Point{ left, top }, cv::Point{ right, bottom }, cv::Scalar{ .0, .0, .0 }, 2);

        // Augment object with some key data.
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", boundingBox.boxID, static_cast<int>(boundingBox.lidarPoints.size()));
        putText(topviewImg, str1, cv::Point2f{ left - 250.0f, bottom + 50.0f }, cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f (m), yw=%2.2f (m)", xwmin, ywmax - ywmin);
        putText(topviewImg, str2, cv::Point2f{ left - 250.0f, bottom + 125.0f }, cv::FONT_ITALIC, 2, currColor);
    }

    // Plot distance markers.
    constexpr int lineSpacing{ 2 };           // gap between distance markers (meter)
    const int nMarkers{ worldSize.height / lineSpacing };
    for (int i{ 0 }; i < nMarkers; ++i) {
        int y{ static_cast<int>((-(i * lineSpacing) * imageSize.height / worldSize.height)) + imageSize.height };
        cv::line(topviewImg, cv::Point{ 0, y }, cv::Point{ imageSize.width, y }, cv::Scalar{ 255.0, .0, .0 });
    }

    // Display image.
    std::string windowName{ "3D Objects" };
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::imshow(windowName, topviewImg);
    cv::waitKey(0);     // Wait for key to be pressed.
}



// https://docs.opencv.org/4.1.0/db/d39/classcv_1_1DescriptorMatcher.html#a0f046f47b68ec7074391e1e85c750cba
void matchBoundingBoxes(const std::vector<cv::DMatch>& curr_KPMatches,
    const std::vector<cv::KeyPoint>& prev_KPs, const std::vector<cv::KeyPoint>& curr_KPs,
    const std::vector<BoundingBox>& prev_BBoxes, const std::vector<BoundingBox>& curr_BBoxes, 
    std::map<int, int>& bbBestMatches) {

    using prevBBoxID2Count = std::map<int, int>;            
    using currBBoxID2Map = std::map<int, prevBBoxID2Count>; 
    currBBoxID2Map matrix_2D;       // map : (currBBoxID, prevBBoxID) -> #(KPMatches in pair of BBoxes)

    for (const auto& match : curr_KPMatches) {

        const cv::KeyPoint curr_KP{ curr_KPs[match.trainIdx] };  // prevFrame.keypoints is indexed by queryIdx.
        const cv::KeyPoint prev_KP{ prev_KPs[match.queryIdx] };  // currFrame.keypoints is indexed by trainIdx.

        for (const auto curr_BBox : curr_BBoxes) {
            if (curr_BBox.roi.contains(curr_KP.pt) == false)      continue;

            for (const auto prev_BBox : prev_BBoxes) {
                if (prev_BBox.roi.contains(prev_KP.pt) == false)  continue;

                matrix_2D[curr_BBox.boxID][prev_BBox.boxID] += 1;
            }
        }
    }

    auto findKeyWithLargestValue{ [](std::map<int, int>& sampleMap) {

            std::pair<int, int> entryWithMaxValue{ std::make_pair(0, 0) };

            for (auto it{ sampleMap.begin() }; it != sampleMap.end(); ++it)
                if (it->second > entryWithMaxValue.second)
                    entryWithMaxValue = std::make_pair(it->first, it->second);

            return entryWithMaxValue.first;
        }
    };

    for (auto currBBoxID{ matrix_2D.begin() }; currBBoxID != matrix_2D.end(); ++currBBoxID)
        bbBestMatches.insert({ findKeyWithLargestValue(currBBoxID->second), currBBoxID->first});
}

// Another method (slower than the one above)
void matchBoundingBoxes_2(const std::vector<cv::DMatch>& curr_KPMatches,
    const std::vector<cv::KeyPoint>& prev_KPs, const std::vector<cv::KeyPoint>& curr_KPs,
    const std::vector<BoundingBox>& prev_BBoxes, const std::vector<BoundingBox>& curr_BBoxes,
    std::map<int, int>& bbBestMatches) {

    const int num_prevBBoxes{ static_cast<int>(prev_BBoxes.size()) };
    const int num_currBBoxes{ static_cast<int>(curr_BBoxes.size()) };


    std::vector<std::vector<float>> num_KPsInPairOfBBoxes(num_currBBoxes, std::vector<float>(num_prevBBoxes, 0));

    for (const auto& match : curr_KPMatches) {

        // prevFrame.keypoints is indexed by queryIdx
        // currFrame.keypoints is indexed by trainIdx
        const cv::KeyPoint currKP{ curr_KPs[match.trainIdx] };
        const cv::KeyPoint prevKP{ prev_KPs[match.queryIdx] };

        std::set<int> currBoundingBoxID, prevBoundingBoxID;

        // for each bounding box in the current frame
        for (const auto boundingBox : curr_BBoxes) {
            if (boundingBox.roi.contains(currKP.pt))
                currBoundingBoxID.insert(boundingBox.boxID);
        }

        // for each bounding box in the previous frame
        for (const auto boundingBox : prev_BBoxes)
            if (boundingBox.roi.contains(prevKP.pt))
                prevBoundingBoxID.insert(boundingBox.boxID);

        for (auto i : currBoundingBoxID)
            for (auto j : prevBoundingBoxID)
                num_KPsInPairOfBBoxes[i][j] += 1;
    }

    for (int currBBox_ID{ 0 }; currBBox_ID < num_currBBoxes; ++currBBox_ID) {
        int num_KPs{ 0 };
        int prevBBox_ID{ -1 };
        for (int it{ 0 }; it < num_prevBBoxes; ++it) {
            if (num_KPs < num_KPsInPairOfBBoxes[currBBox_ID][it]) {
                num_KPs = num_KPsInPairOfBBoxes[currBBox_ID][it];
                prevBBox_ID = it;
            }
        }

        if (prevBBox_ID == -1)     continue;

        // Note the order!
        bbBestMatches.insert({ prevBBox_ID, currBBox_ID });
    }
}


void matchBoundingBoxes___(const DataFrame& prevFrame, const DataFrame& currFrame, std::map<int, int>& bbBestMatches) {

    const int num_currBBoxes{ static_cast<int>(currFrame.getBoundingBoxes().size()) };
    const int num_prevBBoxes{ static_cast<int>(prevFrame.getBoundingBoxes().size()) };

    std::vector<std::vector<float>> num_KPsInPairOfBBoxes(num_currBBoxes, std::vector<float>(num_prevBBoxes, 0));

    for (const auto& match : currFrame.getKPMatches()) {

        // prevFrame.keypoints is indexed by queryIdx
        // currFrame.keypoints is indexed by trainIdx
        const cv::KeyPoint currKP{ currFrame.getKeypoints()[match.trainIdx] };
        const cv::KeyPoint prevKP{ prevFrame.getKeypoints()[match.queryIdx] };

        std::set<int> currBoundingBoxID, prevBoundingBoxID;

        // for each bounding box in the current frame
        for (const auto boundingBox : currFrame.getBoundingBoxes()) {
            if (boundingBox.roi.contains(currKP.pt))
                currBoundingBoxID.insert(boundingBox.boxID);
        }

        // for each bounding box in the previous frame
        for (const auto boundingBox : prevFrame.getBoundingBoxes())
            if (boundingBox.roi.contains(prevKP.pt))
                prevBoundingBoxID.insert(boundingBox.boxID);

        for (auto i : currBoundingBoxID)
            for (auto j : prevBoundingBoxID)
                num_KPsInPairOfBBoxes[i][j] += 1;
    }

    for (int currBBox_ID{ 0 }; currBBox_ID < num_currBBoxes; ++currBBox_ID) {
        int num_KPs{ 0 };
        int prevBBox_ID{ -1 };
        for (int it{ 0 }; it < num_prevBBoxes; ++it) {
            if (num_KPs < num_KPsInPairOfBBoxes[currBBox_ID][it]) {
                num_KPs = num_KPsInPairOfBBoxes[currBBox_ID][it];
                prevBBox_ID = it;
            }
        }

        if (prevBBox_ID == -1)     continue;

        // Note the order!
        bbBestMatches.insert({ prevBBox_ID, currBBox_ID });
    }
}








std::pair<double, double> computeTTCLidar(const std::vector<LidarPoint>& lidarPointsPrev,
    const std::vector<LidarPoint>& lidarPointsCurr, const double frameRate, double& TTC) {

    std::vector<double> prevXvalues, currXvalues;

    for (const auto& prevLidarPoint : lidarPointsPrev)
        prevXvalues.push_back(prevLidarPoint.x);

    for (const auto& currLidarPoint : lidarPointsCurr)
        currXvalues.push_back(currLidarPoint.x);


    // Use medean value.
    std::nth_element(prevXvalues.begin(), prevXvalues.begin() + prevXvalues.size() / 2, prevXvalues.end());
    std::nth_element(currXvalues.begin(), currXvalues.begin() + currXvalues.size() / 2, currXvalues.end());
    const double dist_prev{ prevXvalues[prevXvalues.size() / 2] };
    const double dist_curr{ currXvalues[currXvalues.size() / 2] };

    
    const double dT{ 1 / frameRate };
    TTC = (dist_curr * dT) / (dist_prev - dist_curr);


    std::cout << "\tdistance to previous frame object: " << dist_prev
        << "\n\tdistance to current frame object: " << dist_curr
        << "\n\t(based on Lidar) TTC = " << TTC << '\n';

    return { dist_prev, dist_curr };
}





// Associate a given bounding box with the keypoints it contains.
void clusterKptMatchesWithROI(
    const std::vector<cv::KeyPoint>& kptsPrev, const std::vector<cv::KeyPoint>& kptsCurr,
    const std::vector<cv::DMatch>& kptMatches, BoundingBox* const boundingBox) {

    std::vector<cv::DMatch> KPsMatchesWithROI;

    // Check if the matching keypoints are within the ROI in the camera image.
    for (const auto& kptMatch : kptMatches) {
        cv::KeyPoint train{ kptsCurr[kptMatch.trainIdx] };
        const auto train_pt{ cv::Point{static_cast<int>(train.pt.x), static_cast<int>(train.pt.y) } };

        cv::KeyPoint query{ kptsPrev[kptMatch.queryIdx] };
        const auto query_pt{ cv::Point{ static_cast<int>(query.pt.x), static_cast<int>(query.pt.y) } };

        // Choose good ones.
        if (boundingBox->roi.contains(train_pt) && boundingBox->roi.contains(query_pt))
            KPsMatchesWithROI.push_back(kptMatch);
    }



    // Eliminate outliers by computing the mean of all the euclidean distances between keypoint matches.

    std::vector<float> distances;
    for (const auto& KPsMatch : KPsMatchesWithROI) {
        float normL2{ static_cast<float>(cv::norm(kptsCurr[KPsMatch.trainIdx].pt - kptsPrev[KPsMatch.queryIdx].pt)) };
        distances.push_back(normL2);
    }
    
    //const float mean_distance{ std::accumulate(distances.begin(), distances.end(), 0.0f) / distances.size() };
    std::nth_element(distances.begin(), distances.begin() + distances.size() / 2, distances.end());
    float median_distance{ distances[distances.size() / 2] };

    // Remove matches that are too far away from the mean.
    constexpr float rate{ 1.8f };
    const float upperLimit{ rate * median_distance };
    for (int it{ 0 }; it != distances.size(); ++it) {
        if (distances[it] < upperLimit) {
            // Populate boundingBox.kptMatches with the good matches.
            boundingBox->kptMatches.push_back(KPsMatchesWithROI[it]);
        }
    }
}




// Compute time-to-collision (TTC) based on keypoint correspondences in successive images.
void computeTTCCamera(
    const std::vector<cv::KeyPoint>& kptsPrev, const std::vector<cv::KeyPoint>& kptsCurr,
    const std::vector<cv::DMatch>& kptMatches, const double frameRate, double& TTC) {

    // Compute distance ratios on every pair of keypoints.
    std::vector<double> distRatios;
    
    for (auto it1{ kptMatches.begin() }; it1 != kptMatches.end() - 1; ++it1) {
        const cv::KeyPoint kpOuterCurr{ kptsCurr[it1->trainIdx] };        // kptsCurr is indexed by trainIdx.
        const cv::KeyPoint kpOuterPrev{ kptsPrev[it1->queryIdx] };        // kptsPrev is indexed by queryIdx.

        for (auto it2{ it1 + 1 }; it2 != kptMatches.end(); ++it2) {
            const cv::KeyPoint kpInnerCurr{ kptsCurr[it2->trainIdx] };    // kptsCurr is indexed by trainIdx.
            const cv::KeyPoint kpInnerPrev{ kptsPrev[it2->queryIdx] };    // kptsPrev is indexed by queryIdx.

            // Calculate the current and previous Euclidean distances 
            // between each keypoint in the pair.
            const double distCurr{ cv::norm(kpOuterCurr.pt - kpInnerCurr.pt) };
            const double distPrev{ cv::norm(kpOuterPrev.pt - kpInnerPrev.pt) };

            // Threshold the calculated distRatios by requiring a minimum current distance between keypoints. 
            constexpr double minDist{ 100.0 };

            // Avoid division by zero and apply the threshold.
            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist) {
                const double distRatio{ distCurr / distPrev };
                distRatios.push_back(distRatio);
            }
        }
    }

    // Only continue if the vector of distRatios is not empty.
    if (distRatios.size() == 0) {
        TTC = std::numeric_limits<double>::quiet_NaN();
        std::cout << "\tUse " << kptMatches.size() << " keypoints to compute TTC based on camera."
            << "\n\tNo proper keypoint in a bounding box in front to get a value (distance ratio)"
            << "\n\t(based on Camera) TTC = " << TTC << std::endl;
        return;
    }

    
    // Use the median as a reasonable method of excluding outliers.
    std::nth_element(distRatios.begin(), distRatios.begin() + distRatios.size() / 2, distRatios.end());
    const double median_DISTRatio{ distRatios[distRatios.size() / 2] };


    // Calculate a TTC estimate based on 2D camera features.
    const double dT{ 1 / frameRate };
    TTC = -dT / (1 - median_DISTRatio);
    std::cout << "\tUse " << kptMatches.size() << " keypoints to compute TTC based on camera." 
        << "\n\t" << distRatios.size() << " pairs of keypoints in a bounding box in front to get a value(distance ratio)"
        << "\n\t(based on Camera) TTC = " << TTC << std::endl;
}



