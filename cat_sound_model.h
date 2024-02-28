#ifndef CAT_SOUND_MODEL_H_
#define CAT_SOUND_MODEL_H_

constexpr int kMaxAudioSampleSize = 512;
constexpr int kAudioSampleFrequency = 8000;
constexpr int kFeatureSize = 13;
constexpr int kFeatureCount = 56;
constexpr int kFeatureElementCount = 728;
constexpr int kFeatureStrideMs = 32;
constexpr int kFeatureDurationMs = 64;

constexpr int kCategoryCount = 3;
constexpr const char* kCategoryLabels[kCategoryCount] = {
    "brushing",
    "isolation",
    "food",
};

#endif  // CAT_SOUND_MODEL_H_
