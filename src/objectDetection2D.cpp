#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "objectDetection2D.hpp"




// detects objects in an image using the YOLO library and a set of pre-trained objects from the COCO database;
// a set of 80 classes is listed in "coco.names" and pre-trained weights are stored in "yolov3.weights"
void detectObjects(const cv::Mat& img, const float confThreshold, const float nmsThreshold, 
    const std::string classesFile, const std::string modelConfiguration, 
    const std::string modelWeights, const bool bVis, std::vector<BoundingBox>& bBoxes) {

    // Load class names from file.
    std::vector<std::string> classes;
    std::ifstream ifs(classesFile.c_str());
    std::string line;
    while (getline(ifs, line))
        classes.push_back(line);

    // Load neural network.
    cv::dnn::Net net{ cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights) };
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // Generate 4D blob from input image.
    cv::Mat blob;
    std::vector<cv::Mat> netOutput;
    constexpr double scalefactor{ 1 / 255.0 };
    cv::Size size{ 416, 416 };
    cv::Scalar mean{ .0, .0, .0 };
    constexpr bool swapRB{ false };
    constexpr bool crop{ false };
    cv::dnn::blobFromImage(img, blob, scalefactor, size, mean, swapRB, crop);

    // Get names of output layers.
    std::vector<cv::String> names;
    std::vector<int> outLayers{ net.getUnconnectedOutLayers() };    // Get indices of output layers, 
                                                                    // i.e., layers with unconnected outputs.
    std::vector<cv::String> layersNames{ net.getLayerNames() };     // Get names of all layers in the network.

    names.resize(outLayers.size());
    for (size_t i = 0; i < outLayers.size(); ++i)   // Get the names of the output layers in names
        names[i] = layersNames[outLayers[i] - 1];

    // Invoke forward propagation through network.
    net.setInput(blob);
    net.forward(netOutput, names);

    // Scan through all bounding boxes and keep only the ones with high confidence.
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (const auto& mat : netOutput) {

        //float* data{ (float*)mat.data };
        void* temp{ mat.data };
        float* data{ static_cast<float*>(temp) };

        for (int j{ 0 }; j < mat.rows; ++j, data += mat.cols) {
            cv::Mat scores{ mat.row(j).colRange(5, mat.cols) };
            cv::Point classId;
            double confidence;

            // Get the value and location of the MAXIMUM score.
            cv::minMaxLoc(scores, nullptr, &confidence, nullptr, &classId);
            //cv::minMaxLoc(scores, 0, &confidence, 0, &classId);
            if (confidence > confThreshold) {
                cv::Rect box; 
                int cx, cy;
                cx = static_cast<int>((data[0] * img.cols));
                cy = static_cast<int>((data[1] * img.rows));

                box.width = static_cast<int>((data[2] * img.cols));
                box.height = static_cast<int>((data[3] * img.rows));
                box.x = cx - box.width / 2;     // left
                box.y = cy - box.height / 2;    // top
                boxes.push_back(box);

                classIds.push_back(classId.x);
                confidences.push_back(static_cast<float>(confidence));
            }
        }
    }



    // Perform non-maxima suppression.
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);


    //Run-Time Check Failure #3 - The variable 'bBox' is being used without being initialized.

    for (const int index : indices) {
        BoundingBox bBox;
        bBox.roi = boxes[index];
        bBox.classID = classIds[index];
        bBox.confidence = confidences[index];
        bBox.boxID = static_cast<int>(bBoxes.size());   // zero-based unique identifier for this bounding box

        bBoxes.push_back(bBox);
    }

    // Show results.
    if (bVis) {
        cv::Mat visImg{ img.clone() };

        for (const auto& bBox : bBoxes) {

            // Draw rectangle displaying the bounding box.
            int top, left, width, height;
            top = bBox.roi.y;
            left = bBox.roi.x;
            width = bBox.roi.width;
            height = bBox.roi.height;
            cv::rectangle(visImg, cv::Point{ left, top }, cv::Point{ left + width, top + height }, cv::Scalar{ .0, 255.0, .0 }, 2);

            std::string label{ cv::format("%.2f", bBox.confidence) };
            label = classes[bBox.classID] + ":" + label;

            // Display label at the top of the bounding box
            int baseLine;
            cv::Size labelSize{ getTextSize(label, cv::FONT_ITALIC, 0.5, 1, &baseLine) };
            top = std::max(top, labelSize.height);
            rectangle(visImg,
                cv::Point{ left, top - static_cast<int>(round(1.5 * labelSize.height)) },
                cv::Point{ left + static_cast<int>(round(1.5 * labelSize.width)), top + baseLine },
                cv::Scalar{ 255.0, 255.0, 255.0 }, cv::FILLED);
            cv::putText(visImg, label, cv::Point{ left, top },
                cv::FONT_ITALIC, 0.75, cv::Scalar{ .0, .0, .0 }, 1);
        }

        std::string windowName{ "Object classification" };
        cv::namedWindow(windowName, 1);
        cv::imshow(windowName, visImg);
        cv::waitKey(0);                 // Wait for key to be pressed.
    }
}
