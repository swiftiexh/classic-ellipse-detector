#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

namespace VisibilityValidation {

    // Validate ellipse by checking edge coverage along sampled points on the ellipse.
    // Returns coverage in [0,1] via out parameter and true if above threshold.
    bool validateEllipseCoverage(const cv::Mat &edgeImg, const cv::RotatedRect &el, double &coverage, int samples = 360, int searchRadius = 2);

    // Remove detections whose coverage is below minCoverage.
    void filterByVisibility(std::vector<cv::RotatedRect> &els, std::vector<double> &scores, const cv::Mat &edgeImg, double minCoverage = 0.05);

    // Simple NMS by center distance + bounding-box IoU to deduplicate similar ellipses.
    void deduplicateByVisibility(std::vector<cv::RotatedRect> &els, std::vector<double> &scores, double iouThresh = 0.5);

    // Convenience wrapper: first filter by visibility then deduplicate.
    void ValidateAndDedup(const cv::Mat &edgeImg, std::vector<cv::RotatedRect> &els, std::vector<double> &scores, double minCoverage = 0.05, double iouThresh = 0.5);

}
