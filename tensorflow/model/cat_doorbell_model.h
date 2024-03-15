#ifndef CAT_SOUND_MODEL_H_
#define CAT_SOUND_MODEL_H_

constexpr int kMaxAudioSampleSize = 512;
constexpr int kAudioSampleFrequency = 16000;
constexpr int kFeatureSize = 13;
constexpr int kFeatureCount = 272;
constexpr int kFeatureElementCount = 3536;
constexpr int kFeatureStrideMs = 20;
constexpr int kFeatureDurationMs = 32;

constexpr int kCategoryCount = 2;
constexpr const char* kCategoryLabels[kCategoryCount] = {
    "cat",
    "not_cat",
};

#endif  // CAT_SOUND_MODEL_H_
