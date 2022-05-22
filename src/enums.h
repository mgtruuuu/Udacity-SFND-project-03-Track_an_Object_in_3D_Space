#ifndef ENUMS_H_
#define ENUMS_H_


#include <string_view>



enum class Detector {
    SHITOMASI,
    HARRIS,
    FAST,
    BRISK,
    ORB,
    AKAZE,
    SIFT,
};

enum class Descriptor {
    BRISK,
    BRIEF,
    ORB,
    FREAK,
    AKAZE,
    SIFT,
};

enum class Matcher {
    MAT_BF,
    MAT_FLANN,
};

enum class DescriptorOption {
    DES_BINARY,
    DES_HOG,
};

enum class Selector {
    SEL_NN,
    SEL_KNN,
};



std::string_view getDetector(Detector detectorType);
std::string_view getDescriptor(Descriptor descriptorType);
std::string_view getMatcher(Matcher matcherType);
std::string_view getDescriptorOption(DescriptorOption descriptorOptionType);
std::string_view getSelector(Selector selectorType);


#endif