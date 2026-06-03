#include "FLED.h"
#include <numeric>
#include <algorithm>

namespace {

double calculateEllipseConfidence(const cv::RotatedRect& ellipse) {
    double ratio = ellipse.size.height / ellipse.size.width;
    if (ratio > 1) ratio = 1.0 / ratio;
    double shapeScore = std::min(1.0, ratio * 2);
    double sizeScore = std::min(1.0, std::max(0.1, 
        std::sqrt(ellipse.size.width * ellipse.size.height) / 100.0));
    return (shapeScore + sizeScore) / 2.0;
}

bool isValidEllipse(const cv::RotatedRect& ellipse, double minSize = 5.0, double maxSize = 1000.0) {
    double minDim = std::min(ellipse.size.width, ellipse.size.height);
    double maxDim = std::max(ellipse.size.width, ellipse.size.height);
    
    if (minDim < minSize || maxDim > maxSize) return false;
    if (maxDim / minDim > 30) return false;
    if (ellipse.size.width <= 0 || ellipse.size.height <= 0) return false;
    
    return true;
}

void nonMaximumSuppression(std::vector<cv::RotatedRect>& ellipses, 
                           std::vector<double>& scores, 
                           double iouThreshold = 0.5) {
    if (ellipses.empty()) return;
    
    std::vector<int> indices(ellipses.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return scores[a] > scores[b];
    });
    
    std::vector<bool> suppressed(ellipses.size(), false);
    
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        if (suppressed[idx]) continue;
        
        for (size_t j = i + 1; j < indices.size(); ++j) {
            int idx2 = indices[j];
            if (suppressed[idx2]) continue;
            
            cv::RotatedRect& e1 = ellipses[idx];
            cv::RotatedRect& e2 = ellipses[idx2];
            
            double dx = e1.center.x - e2.center.x;
            double dy = e1.center.y - e2.center.y;
            double dist = std::sqrt(dx * dx + dy * dy);
            
            double avgRadius = (e1.size.width + e1.size.height + e2.size.width + e2.size.height) / 8.0;
            
            if (dist < avgRadius * 0.5) {
                suppressed[idx2] = true;
            }
        }
    }
    
    std::vector<cv::RotatedRect> filteredEllipses;
    std::vector<double> filteredScores;
    
    for (size_t i = 0; i < ellipses.size(); ++i) {
        if (!suppressed[i]) {
            filteredEllipses.push_back(ellipses[i]);
            filteredScores.push_back(scores[i]);
        }
    }
    
    ellipses.swap(filteredEllipses);
    scores.swap(filteredScores);
}

} 

void FLED::SetEnhancedParameters(double theta_fsa, double length_fsa, double T_val,
                                  double minConfidence, double minEllipseSize) {
    SetParameters(theta_fsa, length_fsa, T_val);
    _min_confidence = minConfidence;
    _min_ellipse_size = minEllipseSize;
}

void FLED::EnhancedPostProcessing(double confidenceThreshold, double iouThreshold) {
    if (detEllipses.empty()) return;
    
    std::vector<cv::RotatedRect> filtered;
    std::vector<double> filteredScores;
    
    for (size_t i = 0; i < detEllipses.size(); ++i) {
        if (!isValidEllipse(detEllipses[i], _min_ellipse_size, 2000.0)) {
            continue;
        }
        
        double conf = detEllipseScore[i];
        if (conf < confidenceThreshold) {
            continue;
        }
        
        filtered.push_back(detEllipses[i]);
        filteredScores.push_back(detEllipseScore[i]);
    }
    
    detEllipses.swap(filtered);
    detEllipseScore.swap(filteredScores);
    
    nonMaximumSuppression(detEllipses, detEllipseScore, iouThreshold);
}

void FLED::EnhancedPostProcessingWithVisibility(const cv::Mat &edgeImg, 
                                                 double confidenceThreshold, 
                                                 double iouThreshold,
                                                 double minCoverage) {
    if (detEllipses.empty()) return;
    
    std::vector<cv::RotatedRect> filtered;
    std::vector<double> filteredScores;
    
    for (size_t i = 0; i < detEllipses.size(); ++i) {
        if (!isValidEllipse(detEllipses[i], _min_ellipse_size, 2000.0)) {
            continue;
        }
        
        double conf = detEllipseScore[i];
        if (conf < confidenceThreshold) {
            continue;
        }
        
        filtered.push_back(detEllipses[i]);
        filteredScores.push_back(detEllipseScore[i]);
    }
    
    detEllipses.swap(filtered);
    detEllipseScore.swap(filteredScores);
    
    if (!edgeImg.empty()) {
        VisibilityValidation::ValidateAndDedup(edgeImg, detEllipses, detEllipseScore, minCoverage, iouThreshold);
    } else {
        nonMaximumSuppression(detEllipses, detEllipseScore, iouThreshold);
    }
}