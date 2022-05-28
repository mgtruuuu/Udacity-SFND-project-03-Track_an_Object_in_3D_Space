#ifndef ENUMS_H_
#define ENUMS_H_

//#include <cassert>
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



constexpr std::string_view getDetector(Detector detectorType) {
    switch (detectorType) {
    case Detector::SHITOMASI:   return "SHITOMASI";
    case Detector::HARRIS:      return "HARRIS";
    case Detector::FAST:        return "FAST";
    case Detector::BRISK:       return "BRISK";
    case Detector::ORB:         return "ORB";
    case Detector::AKAZE:       return "AKAZE";
    case Detector::SIFT:        return "SIFT";
    //default:                    assert(false, "Wrong Detector type\n");
    }
}

constexpr std::string_view getDescriptor(Descriptor descriptorType) {
    switch (descriptorType) {
    case Descriptor::BRISK:     return "BRISK";
    case Descriptor::BRIEF:     return "BRIEF";
    case Descriptor::ORB:       return "ORB";
    case Descriptor::FREAK:     return "FREAK";
    case Descriptor::AKAZE:     return "AKAZE";
    case Descriptor::SIFT:      return "SIFT";
    //default:                    assert(false, "Wrong Descriptor type\n");
    }
}

constexpr std::string_view getMatcher(Matcher matcherType) {
    switch (matcherType) {
    case Matcher::MAT_BF:       return "MAT_BF";
    case Matcher::MAT_FLANN:    return "MAT_FLANN";
    //default:                    assert(false, "Wrong Matcher type\n");
    }
}

constexpr std::string_view getDescriptorOption(DescriptorOption descriptorOptionType) {
    switch (descriptorOptionType) {
    case DescriptorOption::DES_BINARY:  return "DES_BINARY";
    case DescriptorOption::DES_HOG:     return "DES_HOG";
    //default:                            assert(false, "Wrong DescriptorOption type\n");
    }
}

constexpr std::string_view getSelector(Selector selectorType) {
    switch (selectorType) {
    case Selector::SEL_NN:      return "SEL_NN";
    case Selector::SEL_KNN:     return "SEL_KNN";
    //default:                    assert(false, "Wrong Selector type\n");
    }
}


#endif