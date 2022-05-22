/* INCLUDES FOR THIS PROJECT */
#include <cmath>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"
#include "enums.h"
#include "lidarData.hpp"
#include "matching2D.hpp"
#include "objectDetection2D.hpp"


void loadCalibrationData(cv::Mat& P_rect_00, cv::Mat& R_rect_00, cv::Mat& RT);
void printResult(const Detector detectorType, const Descriptor descriptorType, 
    const std::vector<Result>& results);
bool writeRecordToFile(const std::string file_name,
    const Detector detectorType, const Descriptor descriptorType, const std::vector<Result>& results);


/* MAIN PROGRAM */
int main(int argc, const char* argv[]) {

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    const std::string dataPath{ "../" };

    // camera
    const std::string imgBasePath{ dataPath + "images/" };
    const std::string imgPrefix{ "KITTI/2011_09_26/image_02/data/000000" };   // left camera, color
    const std::string imgFileType{ ".png" };
    constexpr int imgStartIndex{ 0 };   // first file index to load 
                                        // (Assumes Lidar and camera names have identical naming convention)
    constexpr int imgEndIndex{ 18 };    // last file index to load
    constexpr int imgStepWidth{ 1 };
    constexpr int imgFillWidth{ 4 };    // no. of digits which make up the file index (e.g. img-0001.png)

    // object detection
    const std::string yoloBasePath{ dataPath + "dat/yolo/" };
    const std::string yoloClassesFile{ yoloBasePath + "coco.names" };
    const std::string yoloModelConfiguration{ yoloBasePath + "yolov3.cfg" };
    const std::string yoloModelWeights{ yoloBasePath + "yolov3.weights" };

    // Lidar
    const std::string lidarPrefix{ "KITTI/2011_09_26/velodyne_points/data/000000" };
    const std::string lidarFileType{ ".bin" };

    // calibration data for camera and lidar
    cv::Mat P_rect_00{ 3, 4, cv::DataType<double>::type };  // 3x4 projection matrix after rectification
    cv::Mat R_rect_00{ 4, 4, cv::DataType<double>::type };  // 3x3 rectifying rotation to make image planes co-planar
    cv::Mat RT{ 4, 4, cv::DataType<double>::type };         // rotation matrix and translation vector
    loadCalibrationData(P_rect_00, R_rect_00, RT);


    // misc
    constexpr double sensorFrameRate{ 10.0 / imgStepWidth };// frames per second for Lidar and camera
    constexpr int dataBufferSize{ 2 };      // # images which are held in memory (ring buffer) at the same time
    std::deque<DataFrame> dataBuffer;       // list of data frames which are held in memory at the same time
    std::vector<Result> results;
    

    // visualization options
    constexpr bool bVis_YOLO{ false };
    constexpr bool bVis_LPsFromTopView{ false };
    constexpr bool bVis_CameraView{ false };


    //// detection/description option
    constexpr Detector detectorType{ Detector::SIFT };         // SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
    constexpr Descriptor descriptorType{ Descriptor::SIFT };   // FREAK, BRIEF, BRISK, ORB, AKAZE, SIFT
    constexpr Matcher matcherType{ Matcher::MAT_BF };           // MAT_BF, MAT_FLANN
    constexpr DescriptorOption descriptorOptionType{ DescriptorOption::DES_HOG };   // DES_BINARY, DES_HOG
    constexpr Selector selectorType{ Selector::SEL_KNN };       // SEL_NN, SEL_KNN


    ////


    



    /* MAIN LOOP OVER ALL IMAGES */

    for (int imgIndex{ 0 }; imgIndex <= imgEndIndex - imgStartIndex; imgIndex += imgStepWidth) {
        

        /* LOAD IMAGE INTO BUFFER */

        // Assemble filenames for current index.
        std::ostringstream imgNumber;
        imgNumber << std::setfill('0') << std::setw(imgFillWidth) << imgStartIndex + imgIndex;
        const std::string imgFullFilename{ imgBasePath + imgPrefix + imgNumber.str() + imgFileType };

        // Load image from file.
        cv::Mat img{ cv::imread(imgFullFilename) };

        // Push image into data frame buffer.
        DataFrame frame;
        frame.setCameraImg(img);
        if (dataBuffer.size() > dataBufferSize) 
            dataBuffer.pop_front();

        dataBuffer.push_back(frame);

        std::cout << "#1 : LOAD IMAGE INTO BUFFER done" << std::endl;

        ////




        const auto it_curr{ dataBuffer.end() - 1 };        

        std::vector<BoundingBox> boundingBoxes;



        /* DETECT & CLASSIFY OBJECTS */
        
        constexpr float confThreshold{ 0.2f };
        constexpr float nmsThreshold{ 0.4f };
        detectObjects(it_curr->getCameraImg(), confThreshold, nmsThreshold,
            yoloClassesFile, yoloModelConfiguration, yoloModelWeights, bVis_YOLO, boundingBoxes);
        
        std::cout << "#2 : DETECT & CLASSIFY OBJECTS done" << std::endl;

        ////



        /* CROP LIDAR POINTS */

        // Load 3D Lidar points from file.
        std::vector<LidarPoint> lidarPoints;
        const std::string lidarFullFilename{ imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType };
        loadLidarFromFile(lidarFullFilename, lidarPoints);
        

        // Remove Lidar points based on distance properties.

        // Focus on ego lane.
        constexpr float minZ{ -1.5f };
        constexpr float maxZ{ -0.9f };
        constexpr float minX{ 2.0f };
        constexpr float maxX{ 20.0f };
        constexpr float maxY{ 2.0f };
        constexpr float minR{ 0.1f };    
        cropLidarPoints(minX, maxX, maxY, minZ, maxZ, minR, lidarPoints);

        // Push lidarpoints for current frame to end of data buffer.
        it_curr->setLidarPoints(lidarPoints);

        std::cout << "#3 : CROP LIDAR POINTS done" << std::endl;

        ////



        /* CLUSTER LIDAR POINT CLOUD */

        // Associate Lidar points with camera-based ROI.
        constexpr float shrinkFactor{ 0.1f };   // Shrinks each bounding box by the given percentage (length ratio)
                                                // to avoid 3D object merging at the edges of an ROI.
        clusterLidarWithROI(it_curr->getLidarPoints(), P_rect_00, R_rect_00, RT, shrinkFactor,
            boundingBoxes);

        // Visualize 3D objects.
        if (bVis_LPsFromTopView)
            show3DObjects(boundingBoxes, cv::Size{ 4, 20 }, cv::Size{ 2000, 2000 });

        std::cout << "#4 : CLUSTER LIDAR POINT CLOUD done" << std::endl;

        ////



        /* DETECT IMAGE KEYPOINTS */
        /* EXTRACT KEYPOINT DESCRIPTORS */

        // Convert current image to grayscale.
        cv::Mat imgGray;
        cv::cvtColor(it_curr->getCameraImg(), imgGray, cv::COLOR_BGR2GRAY);

        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        std::vector<cv::DMatch> matches;
        std::map<int, int> bbBestMatches;

        
        detectKeypoints(detectorType, imgGray, keypoints);

        computeDescriptors(detectorType, descriptorType, imgGray, keypoints, descriptors);

        // Push keypoints and descriptor for current frame to end of data buffer.
        it_curr->setKeypoints(keypoints);
        it_curr->setDescriptors(descriptors);

        // Define this fcn. just to fill in frame.boundingBoxes[i].keypoints for all i.
        clusterKPsWithROI(keypoints, shrinkFactor, boundingBoxes);

        


        std::cout << "#5 : DETECT KEYPOINTS done" << std::endl;
        std::cout << "#6 : EXTRACT DESCRIPTORS done" << std::endl;

        ////
        ////
        


        /* MATCH KEYPOINT DESCRIPTORS */
        
        // Wait until at least two images have been processed.
        if (dataBuffer.size() <= 1) {

            // Push empty data members for the first frame (only) to end of data buffer.
            it_curr->setKPMatches(std::vector<cv::DMatch>());
            it_curr->setBBMatches(std::map<int, int>());
            it_curr->setBoundingBoxes(boundingBoxes);

            continue;
        }





        const auto it_prev{ it_curr - 1 };
        constexpr bool crossCheck{ false };
        matchDescriptors(it_prev->getDescriptors(), it_curr->getDescriptors(),
            matcherType, descriptorOptionType, selectorType, crossCheck, matches);

        // Push kptMatches for current frame to end of data buffer.
        it_curr->setKPMatches(matches);

        std::cout << "#7 : MATCH KEYPOINT DESCRIPTORS done" << std::endl;

        ////



        /* TRACK 3D OBJECT BOUNDING BOXES */

        //// TASK FP.1 -> Match list of 3D objects (vector<BoundingBox*>) 
        ////              between current and previous frame (implement ->matchBoundingBoxes)
        
        // Associate bounding boxes between current and previous frame using keypoint matches.
        matchBoundingBoxes(it_curr->getKPMatches(), it_prev->getKeypoints(), it_curr->getKeypoints(),
            it_prev->getBoundingBoxes(), boundingBoxes, bbBestMatches);
                            
        // Push bbMatches for current frame to end of data buffer.
        it_curr->setBBMatches(bbBestMatches);

        std::cout << "#8 : TRACK 3D OBJECT BOUNDING BOXES done" << std::endl;

        ////



        /* COMPUTE TTC ON OBJECT IN FRONT */

        // Loop over all BB match pairs.
        for (const auto& pairOfBBoxIDs : it_curr->getBBMatches()) {

            // Find bounding boxes associates with current match.
            const BoundingBox* prevBB{ nullptr };
            BoundingBox* currBB{ nullptr };

            for (const auto& prevBBox : it_prev->getBoundingBoxes()) {
                if (pairOfBBoxIDs.first == prevBBox.boxID) {    // Check wether current match partner corresponds to this BB.
                    prevBB = &prevBBox;
                    break;
                }
            }

            for (auto& currBBox : boundingBoxes) {
                if (pairOfBBoxIDs.second == currBBox.boxID) {   // Check wether current match partner corresponds to this BB.
                    currBB = &currBBox;
                    break;
                }
            }


            if (prevBB == nullptr || currBB == nullptr)
                assert(false, "Something wrong...\n");


            // Compute TTC for current match.

            // Only compute TTC if we have Lidar points.
            if (prevBB->lidarPoints.size() == 0 || currBB->lidarPoints.size() == 0)
                continue;



            double ttcLidar;
            std::pair<double, double> pair_distance{
                computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttcLidar)
            };

            double ttcCamera;
            clusterKptMatchesWithROI(it_prev->getKeypoints(), 
                it_curr->getKeypoints(), it_curr->getKPMatches(), currBB);
            computeTTCCamera(it_prev->getKeypoints(), it_curr->getKeypoints(), currBB->kptMatches, 
                sensorFrameRate, ttcCamera);


            // For Performance Evaluation
            {
                if (imgIndex == imgStepWidth) {
                    Result result_prev;
                    result_prev.num_LPsInBB = static_cast<int>(prevBB->lidarPoints.size());
                    result_prev.lidarDist = pair_distance.first;
                    result_prev.ttcLidar = -1.0;
                    result_prev.ttcCamera = -1.0;

                    results.push_back(result_prev);
                }

                it_curr->result.num_LPsInBB = static_cast<int>(currBB->lidarPoints.size());
                it_curr->result.lidarDist = pair_distance.second;
                it_curr->result.ttcLidar = ttcLidar;
                it_curr->result.ttcCamera = ttcCamera;

                results.push_back(it_curr->result);
            }


            if (bVis_CameraView) {
                cv::Mat visImg{ it_curr->getCameraImg().clone()};
                //showLidarTopview(currBB->lidarPoints, cv::Size(4.0, 20.0), cv::Size(2000, 2000), true);
                showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00, R_rect_00, RT, &visImg, true);
                cv::rectangle(visImg, cv::Point{ currBB->roi.x, currBB->roi.y },
                    cv::Point{ currBB->roi.x + currBB->roi.width, currBB->roi.y + currBB->roi.height }, cv::Scalar{ .0, 255.0, .0 }, 2);

                char str[200];
                sprintf(str, "TTC Lidar : %f s, TTC Camera : %f s", ttcLidar, ttcCamera);
                putText(visImg, str, cv::Point2f{ 80.0f, 50.0f }, cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar{ .0, .0, 255.0 });

                const std::string windowName{ "Final Results : TTC" };
                cv::namedWindow(windowName, 4);
                cv::imshow(windowName, visImg);
                std::cout << "Press key to continue to next frame" << std::endl;
                cv::waitKey(0);
            }

        }   // eof loop over all BB matches         


        // Push boundingBoxes for current frame to end of data buffer.
        it_curr->setBoundingBoxes(boundingBoxes);

    }    // eof loop over all images


    printResult(detectorType, descriptorType, results);

    //if (writeRecordToFile("kkk.txt", detectorType, descriptorType, results) == false)
    //    assert(false, "Check the fcn. writeRecordToFile()\n");

    return 0;
}






void loadCalibrationData(cv::Mat& P_rect_00, cv::Mat& R_rect_00, cv::Mat& RT) {
    RT.at<double>(0, 0) = 7.533745e-03; RT.at<double>(0, 1) = -9.999714e-01; RT.at<double>(0, 2) = -6.166020e-04; RT.at<double>(0, 3) = -4.069766e-03;
    RT.at<double>(1, 0) = 1.480249e-02; RT.at<double>(1, 1) = 7.280733e-04; RT.at<double>(1, 2) = -9.998902e-01; RT.at<double>(1, 3) = -7.631618e-02;
    RT.at<double>(2, 0) = 9.998621e-01; RT.at<double>(2, 1) = 7.523790e-03; RT.at<double>(2, 2) = 1.480755e-02; RT.at<double>(2, 3) = -2.717806e-01;
    RT.at<double>(3, 0) = 0.0; RT.at<double>(3, 1) = 0.0; RT.at<double>(3, 2) = 0.0; RT.at<double>(3, 3) = 1.0;

    R_rect_00.at<double>(0, 0) = 9.999239e-01; R_rect_00.at<double>(0, 1) = 9.837760e-03; R_rect_00.at<double>(0, 2) = -7.445048e-03; R_rect_00.at<double>(0, 3) = 0.0;
    R_rect_00.at<double>(1, 0) = -9.869795e-03; R_rect_00.at<double>(1, 1) = 9.999421e-01; R_rect_00.at<double>(1, 2) = -4.278459e-03; R_rect_00.at<double>(1, 3) = 0.0;
    R_rect_00.at<double>(2, 0) = 7.402527e-03; R_rect_00.at<double>(2, 1) = 4.351614e-03; R_rect_00.at<double>(2, 2) = 9.999631e-01; R_rect_00.at<double>(2, 3) = 0.0;
    R_rect_00.at<double>(3, 0) = 0; R_rect_00.at<double>(3, 1) = 0; R_rect_00.at<double>(3, 2) = 0; R_rect_00.at<double>(3, 3) = 1;

    P_rect_00.at<double>(0, 0) = 7.215377e+02; P_rect_00.at<double>(0, 1) = 0.000000e+00; P_rect_00.at<double>(0, 2) = 6.095593e+02; P_rect_00.at<double>(0, 3) = 0.000000e+00;
    P_rect_00.at<double>(1, 0) = 0.000000e+00; P_rect_00.at<double>(1, 1) = 7.215377e+02; P_rect_00.at<double>(1, 2) = 1.728540e+02; P_rect_00.at<double>(1, 3) = 0.000000e+00;
    P_rect_00.at<double>(2, 0) = 0.000000e+00; P_rect_00.at<double>(2, 1) = 0.000000e+00; P_rect_00.at<double>(2, 2) = 1.000000e+00; P_rect_00.at<double>(2, 3) = 0.000000e+00;
}


void printResult(const Detector detectorType, const Descriptor descriptorType, const std::vector<Result>& results) {
    std::cout << "Detector" << '\t' << "Descriptor" << '\t'
        << "#LPs" << '\t' << "DIST(LP)" << '\t' << "TTC(LP)" << '\t' << "TTC(Camera)" << '\n';
    for (const auto& result : results) {
        std::cout
            << static_cast<std::string>(getDetector(detectorType)) << '\t'
            << static_cast<std::string>(getDescriptor(descriptorType)) << '\t'
            << result.num_LPsInBB << '\t'
            << result.lidarDist << '\t'
            << result.ttcLidar << '\t'
            << result.ttcCamera << '\n';
    }
}

bool writeRecordToFile(const std::string file_name,
    const Detector detectorType, const Descriptor descriptorType, const std::vector<Result>& results) {

    std::ofstream file;
    file.open(file_name, std::ios_base::app);
    for (const auto& result : results) {
        file << static_cast<std::string>(getDetector(detectorType)) << '\t'
            << static_cast<std::string>(getDescriptor(descriptorType)) << '\t'
            << result.num_LPsInBB << '\t'
            << result.lidarDist << '\t'
            << result.ttcLidar << '\t'
            << result.ttcCamera << '\n';
    }
    file.close();

    return true;
}