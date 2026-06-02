#include "VisibilityValidation.h"

namespace VisibilityValidation {

bool validateEllipseCoverage(const cv::Mat &edgeImg, const cv::RotatedRect &el, double &coverage, int samples, int searchRadius)
{
    if (edgeImg.empty()) { coverage = 0.0; return false; }
    const int rows = edgeImg.rows, cols = edgeImg.cols;
    const double angle = el.angle * CV_PI / 180.0;
    const double a = el.size.width / 2.0;
    const double b = el.size.height / 2.0;
    const cv::Point2f center = el.center;

    int hits = 0, valid = 0;
    for (int i = 0; i < samples; ++i) {
        double theta = 2.0 * CV_PI * i / samples;
        double x = a * cos(theta);
        double y = b * sin(theta);
        // rotate
        double xr = x * cos(angle) - y * sin(angle) + center.x;
        double yr = x * sin(angle) + y * cos(angle) + center.y;
        int xi = int(round(xr)), yi = int(round(yr));
        if (xi < 0 || yi < 0 || xi >= cols || yi >= rows) continue;
        valid++;
        bool found = false;
        for (int dy = -searchRadius; dy <= searchRadius && !found; ++dy) {
            for (int dx = -searchRadius; dx <= searchRadius; ++dx) {
                int nx = xi + dx, ny = yi + dy;
                if (nx < 0 || ny < 0 || nx >= cols || ny >= rows) continue;
                if (edgeImg.at<uchar>(ny, nx) != 0) { found = true; break; }
            }
        }
        if (found) hits++;
    }
    coverage = valid > 0 ? double(hits) / double(valid) : 0.0;
    return coverage > 0.0;
}

void filterByVisibility(std::vector<cv::RotatedRect> &els, std::vector<double> &scores, const cv::Mat &edgeImg, double minCoverage)
{
    std::vector<cv::RotatedRect> outEls;
    std::vector<double> outScores;
    outEls.reserve(els.size()); outScores.reserve(scores.size());
    for (size_t i = 0; i < els.size(); ++i) {
        double cov = 0.0;
        validateEllipseCoverage(edgeImg, els[i], cov);
        if (cov >= minCoverage) {
            outEls.push_back(els[i]);
            outScores.push_back(scores[i]);
        }
    }
    els.swap(outEls);
    scores.swap(outScores);
}

static double rectIoU(const cv::Rect2f &A, const cv::Rect2f &B)
{
    float x1 = std::max(A.x, B.x);
    float y1 = std::max(A.y, B.y);
    float x2 = std::min(A.x + A.width, B.x + B.width);
    float y2 = std::min(A.y + A.height, B.y + B.height);
    if (x2 <= x1 || y2 <= y1) return 0.0;
    double inter = double(x2 - x1) * double(y2 - y1);
    double uni = double(A.width) * double(A.height) + double(B.width) * double(B.height) - inter;
    return inter / uni;
}

void deduplicateByVisibility(std::vector<cv::RotatedRect> &els, std::vector<double> &scores, double iouThresh)
{
    if (els.empty()) return;
    // sort indices by score desc
    std::vector<int> idx(els.size());
    for (int i = 0; i < (int)idx.size(); ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return scores[a] > scores[b]; });

    std::vector<char> keep(els.size(), 0);
    for (size_t _i = 0; _i < idx.size(); ++_i) {
        int i = idx[_i];
        if (keep[i]) continue; // already kept
        keep[i] = 1;
        cv::Rect2f Ai = els[i].boundingRect2f();
        for (size_t _j = _i + 1; _j < idx.size(); ++_j) {
            int j = idx[_j];
            if (keep[j]) continue;
            cv::Rect2f Aj = els[j].boundingRect2f();
            double iou = rectIoU(Ai, Aj);
            // also consider center proximity relative to size
            double dx = els[i].center.x - els[j].center.x;
            double dy = els[i].center.y - els[j].center.y;
            double dist = sqrt(dx*dx + dy*dy);
            double sizeRef = std::min(std::min(els[i].size.width, els[i].size.height), std::min(els[j].size.width, els[j].size.height));
            if (iou > iouThresh || dist < sizeRef * 0.2) {
                // suppress lower score (j)
                keep[j] = 0; // mark suppressed (we'll not move it to output)
                // mark as removed by setting special flag to 2
                keep[j] = 2;
            }
        }
    }
    std::vector<cv::RotatedRect> outEls;
    std::vector<double> outScores;
    for (int i = 0; i < (int)els.size(); ++i) {
        if (keep[i] == 1) {
            outEls.push_back(els[i]);
            outScores.push_back(scores[i]);
        }
    }
    els.swap(outEls); scores.swap(outScores);
}

void ValidateAndDedup(const cv::Mat &edgeImg, std::vector<cv::RotatedRect> &els, std::vector<double> &scores, double minCoverage, double iouThresh)
{
    filterByVisibility(els, scores, edgeImg, minCoverage);
    deduplicateByVisibility(els, scores, iouThresh);
}

} // namespace
