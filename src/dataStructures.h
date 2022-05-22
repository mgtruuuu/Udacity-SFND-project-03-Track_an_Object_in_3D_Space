#ifndef dataStructures_h
#define dataStructures_h

#include <vector>
#include <map>
#include <opencv2/core.hpp>





// single lidar point in space
struct LidarPoint {
    double x, y, z, r;  // x,y,z in [m], r is point reflectivity.
};

// bounding box around a classified object (contains both 2D and 3D data)
struct BoundingBox {

    int boxID;      // unique identifier for this bounding box
    int trackID;    // unique identifier for the track to which this bounding box belongs

    cv::Rect roi;   // 2D region-of-interest in image coordinates
    int classID;    // ID based on class file provided to YOLO framework
    double confidence; // classification trust

    std::vector<LidarPoint> lidarPoints;    // Lidar 3D points which project into 2D image roi
    std::vector<cv::KeyPoint> keypoints;    // keypoints enclosed by 2D roi
    std::vector<cv::DMatch> kptMatches;     // keypoint matches enclosed by 2D roi
};


struct Result {
    int num_LPsInBB;
    double lidarDist;
    double ttcLidar;
    double ttcCamera;
};


// Represents the available sensor information at the same time instance.
class DataFrame {

private:
    cv::Mat cameraImg;                      // camera image

    std::vector<cv::KeyPoint> keypoints;    // 2D keypoints within camera image
    cv::Mat descriptors;                    // keypoint descriptors
    std::vector<cv::DMatch> kptMatches;     // keypoint matches between previous and current frame
    
    std::vector<LidarPoint> lidarPoints;

    std::vector<BoundingBox> boundingBoxes;// ROI around detected objects in 2D image coordinates
    std::map<int, int> bbMatches;           // bounding box matches between previous and current frame

public:
    Result result;
    

public:
    const cv::Mat& getCameraImg() const { return cameraImg; }
    const std::vector<cv::KeyPoint>& getKeypoints() const { return keypoints; }
    const cv::Mat& getDescriptors() const { return descriptors; }
    const std::vector<cv::DMatch>& getKPMatches() const { return kptMatches; }
    const std::vector<LidarPoint>& getLidarPoints() const { return lidarPoints; }
    const std::vector<BoundingBox>& getBoundingBoxes() const { return boundingBoxes; }
    const std::map<int, int>& getBBMatches() const { return bbMatches; }

    void setCameraImg(cv::Mat cameraImg) { this->cameraImg = cameraImg; }
    void setKeypoints(std::vector<cv::KeyPoint> keypoints) { this->keypoints = keypoints; }
    void setDescriptors(cv::Mat descriptors) { this->descriptors = descriptors; }
    void setKPMatches(std::vector<cv::DMatch> kptMatches) { this->kptMatches = kptMatches; }
    void setLidarPoints(std::vector<LidarPoint> lidarPoints) { this->lidarPoints = lidarPoints; }
    void setBoundingBoxes(std::vector<BoundingBox> boundingBoxes) { this->boundingBoxes = boundingBoxes; }
    void setBBMatches(std::map<int, int> bbMatches) { this->bbMatches = bbMatches; }
};


#endif /* dataStructures_h */
