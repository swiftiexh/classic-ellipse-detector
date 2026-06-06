#include "FLED.h"
#include <numeric>
#include <algorithm>
#include <cmath>
#include <vector>
#include <map>

namespace {

double calculateEllipseConfidence(const cv::RotatedRect& ellipse) {
    double ratio = ellipse.size.height / ellipse.size.width;
    if (ratio > 1) ratio = 1.0 / ratio;
    double shapeScore = std::min(1.0, ratio * 2);
    double sizeScore = std::min(1.0, std::max(0.1, 
        std::sqrt(ellipse.size.width * ellipse.size.height) / 100.0));
    return (shapeScore + sizeScore) / 2.0;
}

bool isValidEllipse(const cv::RotatedRect& ellipse, double minSize, double maxSize) {
    double minDim = std::min(ellipse.size.width, ellipse.size.height);
    double maxDim = std::max(ellipse.size.width, ellipse.size.height);
    
    if (minDim < minSize || maxDim > maxSize) return false;
    if (maxDim / minDim > 30) return false;
    if (ellipse.size.width <= 0 || ellipse.size.height <= 0) return false;
    
    return true;
}

double calculateEdgeContinuity(const cv::Mat& edgeImg, const cv::RotatedRect& ellipse, int samplePoints) {
    if (edgeImg.empty()) return 1.0;
    
    int count = 0;
    int validPoints = 0;
    
    for (int i = 0; i < samplePoints; ++i) {
        double angle = 2 * CV_PI * i / samplePoints;
        double x = ellipse.center.x + ellipse.size.width / 2.0 * cos(angle);
        double y = ellipse.center.y + ellipse.size.height / 2.0 * sin(angle);
        
        if (x >= 0 && x < edgeImg.cols && y >= 0 && y < edgeImg.rows) {
            validPoints++;
            if (edgeImg.at<uchar>(cv::Point(x, y)) > 0) {
                count++;
            }
        }
    }
    
    return validPoints > 0 ? static_cast<double>(count) / validPoints : 0.0;
}

double calculateShapeConsistency(const cv::RotatedRect& ellipse) {
    double width = ellipse.size.width;
    double height = ellipse.size.height;
    double ratio = std::min(width, height) / std::max(width, height);
    
    double aspectScore = std::min(1.0, ratio * 4);
    double roundnessScore = 1.0 - std::abs(1.0 - ratio) * 0.5;
    
    return (aspectScore + roundnessScore) / 2.0;
}

void adaptiveThresholdFilter(std::vector<cv::RotatedRect>& ellipses, 
                             std::vector<double>& scores,
                             const cv::Mat& edgeImg) {
    if (ellipses.empty()) return;
    
    std::vector<double> continuityScores;
    double avgContinuity = 0.0;
    
    for (size_t i = 0; i < ellipses.size(); i++) {
        double cont = calculateEdgeContinuity(edgeImg, ellipses[i], 36);
        continuityScores.push_back(cont);
        avgContinuity += cont;
    }
    
    avgContinuity /= ellipses.size();
    double adaptiveThreshold = std::max(0.3, avgContinuity * 0.7);
    
    std::vector<cv::RotatedRect> filtered;
    std::vector<double> filteredScores;
    
    for (size_t i = 0; i < ellipses.size(); i++) {
        double combinedScore = scores[i] * 0.6 + continuityScores[i] * 0.4;
        if (continuityScores[i] >= adaptiveThreshold && combinedScore > 0.3) {
            filtered.push_back(ellipses[i]);
            filteredScores.push_back(combinedScore);
        }
    }
    
    ellipses.swap(filtered);
    scores.swap(filteredScores);
}

void clusterEllipsesByCenter(std::vector<cv::RotatedRect>& ellipses, 
                             std::vector<double>& scores,
                             double clusterDistance) {
    if (ellipses.empty()) return;
    
    std::vector<bool> processed(ellipses.size(), false);
    std::vector<cv::RotatedRect> finalEllipses;
    std::vector<double> finalScores;
    
    for (size_t i = 0; i < ellipses.size(); i++) {
        if (processed[i]) continue;
        
        std::vector<size_t> cluster;
        cluster.push_back(i);
        processed[i] = true;
        
        for (size_t j = i + 1; j < ellipses.size(); j++) {
            if (processed[j]) continue;
            
            double dx = ellipses[i].center.x - ellipses[j].center.x;
            double dy = ellipses[i].center.y - ellipses[j].center.y;
            double dist = std::sqrt(dx * dx + dy * dy);
            
            if (dist < clusterDistance) {
                cluster.push_back(j);
                processed[j] = true;
            }
        }
        
        if (cluster.size() == 1) {
            finalEllipses.push_back(ellipses[i]);
            finalScores.push_back(scores[i]);
        } else {
            int bestIdx = cluster[0];
            double bestScore = scores[cluster[0]];
            
            for (size_t k = 1; k < cluster.size(); k++) {
                if (scores[cluster[k]] > bestScore) {
                    bestScore = scores[cluster[k]];
                    bestIdx = cluster[k];
                }
            }
            
            finalEllipses.push_back(ellipses[bestIdx]);
            finalScores.push_back(bestScore);
        }
    }
    
    ellipses.swap(finalEllipses);
    scores.swap(finalScores);
}

void refineEllipseFit(std::vector<cv::RotatedRect>& ellipses, const cv::Mat& edgeImg) {
    if (ellipses.empty() || edgeImg.empty()) return;
    
    for (size_t i = 0; i < ellipses.size(); i++) {
        std::vector<cv::Point> edgePoints;
        cv::RotatedRect& ellipse = ellipses[i];
        int sampleRadius = std::max(3, static_cast<int>(std::min(ellipse.size.width, ellipse.size.height) / 4));
        
        for (int r = -sampleRadius; r <= sampleRadius; r++) {
            for (int c = -sampleRadius; c <= sampleRadius; c++) {
                int x = static_cast<int>(ellipse.center.x) + c;
                int y = static_cast<int>(ellipse.center.y) + r;
                
                if (x >= 0 && x < edgeImg.cols && y >= 0 && y < edgeImg.rows) {
                    if (edgeImg.at<uchar>(y, x) > 0) {
                        edgePoints.push_back(cv::Point(x, y));
                    }
                }
            }
        }
        
        if (edgePoints.size() >= 6) {
            cv::RotatedRect refined = cv::fitEllipse(edgePoints);
            double dx = refined.center.x - ellipse.center.x;
            double dy = refined.center.y - ellipse.center.y;
            
            if (std::sqrt(dx * dx + dy * dy) < 10) {
                ellipse = refined;
            }
        }
    }
}

void nonMaximumSuppression(std::vector<cv::RotatedRect>& ellipses, 
                           std::vector<double>& scores, 
                           double iouThreshold) {
    if (ellipses.empty()) return;
    
    std::vector<int> indices(ellipses.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return scores[a] > scores[b];
    });
    
    std::vector<bool> suppressed(ellipses.size(), false);
    
    for (size_t i = 0; i < indices.size(); i++) {
        int idx = indices[i];
        if (suppressed[idx]) continue;
        
        for (size_t j = i + 1; j < indices.size(); j++) {
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
    
    for (size_t i = 0; i < ellipses.size(); i++) {
        if (!suppressed[i]) {
            filteredEllipses.push_back(ellipses[i]);
            filteredScores.push_back(scores[i]);
        }
    }
    
    ellipses.swap(filteredEllipses);
    scores.swap(filteredScores);
}

void advancedNonMaximumSuppression(std::vector<cv::RotatedRect>& ellipses,
                                   std::vector<double>& scores,
                                   double iouThreshold) {
    if (ellipses.empty()) return;
    
    std::vector<int> indices(ellipses.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return scores[a] > scores[b];
    });
    
    std::vector<bool> suppressed(ellipses.size(), false);
    
    for (size_t i = 0; i < indices.size(); i++) {
        int idx = indices[i];
        if (suppressed[idx]) continue;
        
        cv::RotatedRect& e1 = ellipses[idx];
        cv::Rect2f bbox1 = e1.boundingRect2f();
        
        for (size_t j = i + 1; j < indices.size(); j++) {
            int idx2 = indices[j];
            if (suppressed[idx2]) continue;
            
            cv::RotatedRect& e2 = ellipses[idx2];
            cv::Rect2f bbox2 = e2.boundingRect2f();
            
            float x1 = std::max(bbox1.x, bbox2.x);
            float y1 = std::max(bbox1.y, bbox2.y);
            float x2 = std::min(bbox1.x + bbox1.width, bbox2.x + bbox2.width);
            float y2 = std::min(bbox1.y + bbox1.height, bbox2.y + bbox2.height);
            
            if (x2 <= x1 || y2 <= y1) continue;
            
            double inter = (x2 - x1) * (y2 - y1);
            double unionArea = bbox1.width * bbox1.height + bbox2.width * bbox2.height - inter;
            double iou = inter / unionArea;
            
            double dx = e1.center.x - e2.center.x;
            double dy = e1.center.y - e2.center.y;
            double dist = std::sqrt(dx * dx + dy * dy);
            double avgRadius = (e1.size.width + e1.size.height + e2.size.width + e2.size.height) / 8.0;
            double sizeDiff = std::abs(e1.size.width - e2.size.width) + std::abs(e1.size.height - e2.size.height);
            
            if (iou > iouThreshold || (dist < avgRadius * 0.3 && sizeDiff < 20)) {
                suppressed[idx2] = true;
            }
        }
    }
    
    std::vector<cv::RotatedRect> filteredEllipses;
    std::vector<double> filteredScores;
    
    for (size_t i = 0; i < ellipses.size(); i++) {
        if (!suppressed[i]) {
            filteredEllipses.push_back(ellipses[i]);
            filteredScores.push_back(scores[i]);
        }
    }
    
    ellipses.swap(filteredEllipses);
    scores.swap(filteredScores);
}

} // end anonymous namespace

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
    
    double avgConfidence = 0.0;
    for (size_t i = 0; i < detEllipses.size(); i++) {
        avgConfidence += detEllipseScore[i];
    }
    avgConfidence /= detEllipses.size();
    
    double adaptiveThreshold = confidenceThreshold;
    if (avgConfidence < 0.4) {
        adaptiveThreshold = std::max(0.001, confidenceThreshold * 0.3);
    } else if (avgConfidence < 0.6) {
        adaptiveThreshold = std::max(0.001, confidenceThreshold * 0.6);
    }
    
    for (size_t i = 0; i < detEllipses.size(); i++) {
        double conf = detEllipseScore[i];
        
        if (conf < adaptiveThreshold) {
            continue;
        }
        
        double width = detEllipses[i].size.width;
        double height = detEllipses[i].size.height;
        if (width < 3 || height < 3) {
            continue;
        }
        if (width > 2000 || height > 2000) {
            continue;
        }
        
        filtered.push_back(detEllipses[i]);
        filteredScores.push_back(detEllipseScore[i]);
    }
    
    detEllipses.swap(filtered);
    detEllipseScore.swap(filteredScores);
    
    if (detEllipses.size() > 1) {
        nonMaximumSuppression(detEllipses, detEllipseScore, iouThreshold);
    }
}

void FLED::EnhancedPostProcessingWithVisibility(double confidenceThreshold, 
                                                 double iouThreshold,
                                                 double minCoverage) {
    if (detEllipses.empty()) return;
    
    std::vector<cv::RotatedRect> filtered;
    std::vector<double> filteredScores;
    
    double avgScore = 0.0;
    for (size_t i = 0; i < detEllipses.size(); i++) {
        avgScore += detEllipseScore[i];
    }
    avgScore /= detEllipses.size();
    
    double adaptiveConfThreshold = confidenceThreshold;
    if (avgScore < 0.5) {
        adaptiveConfThreshold = std::max(0.01, confidenceThreshold * 0.5);
    }
    
    for (size_t i = 0; i < detEllipses.size(); i++) {
        if (!isValidEllipse(detEllipses[i], _min_ellipse_size, 2000.0)) {
            continue;
        }
        
        double conf = detEllipseScore[i];
        if (conf < adaptiveConfThreshold) {
            continue;
        }
        
        if (!imgCanny.empty()) {
            double coverage = calculateEdgeContinuity(imgCanny, detEllipses[i], 48);
            double dynamicCoverageThreshold = minCoverage;
            
            if (conf > 0.8) {
                dynamicCoverageThreshold = minCoverage * 0.5;
            } else if (conf > 0.6) {
                dynamicCoverageThreshold = minCoverage * 0.7;
            }
            
            if (coverage < dynamicCoverageThreshold) {
                continue;
            }
        }
        
        filtered.push_back(detEllipses[i]);
        filteredScores.push_back(detEllipseScore[i]);
    }
    
    detEllipses.swap(filtered);
    detEllipseScore.swap(filteredScores);
    
    if (!detEllipses.empty()) {
        advancedNonMaximumSuppression(detEllipses, detEllipseScore, iouThreshold);
    }
}

void FLED::AdvancedEllipseRefinement(const cv::Mat& edgeImg, bool refineFit, bool enableClustering) {
    if (detEllipses.empty()) return;
    
    if (!edgeImg.empty() && refineFit) {
        refineEllipseFit(detEllipses, edgeImg);
    }
    
    if (enableClustering) {
        clusterEllipsesByCenter(detEllipses, detEllipseScore, 25.0);
    }
}

void FLED::MultiStageFiltering(const cv::Mat& edgeImg, 
                               double initialConfidence,
                               double finalConfidence,
                               double clusterDistance) {
    if (detEllipses.empty()) return;
    
    std::vector<cv::RotatedRect> filtered;
    std::vector<double> filteredScores;
    
    for (size_t i = 0; i < detEllipses.size(); i++) {
        if (!isValidEllipse(detEllipses[i], _min_ellipse_size, 2000.0)) {
            continue;
        }
        
        double shapeScore = calculateShapeConsistency(detEllipses[i]);
        double combinedScore = detEllipseScore[i] * 0.7 + shapeScore * 0.3;
        
        if (combinedScore >= initialConfidence) {
            filtered.push_back(detEllipses[i]);
            filteredScores.push_back(combinedScore);
        }
    }
    
    detEllipses.swap(filtered);
    detEllipseScore.swap(filteredScores);
    
    if (!edgeImg.empty()) {
        adaptiveThresholdFilter(detEllipses, detEllipseScore, edgeImg);
    }
    
    clusterEllipsesByCenter(detEllipses, detEllipseScore, clusterDistance);
    
    std::vector<cv::RotatedRect> finalFiltered;
    std::vector<double> finalScores;
    
    for (size_t i = 0; i < detEllipses.size(); i++) {
        if (detEllipseScore[i] >= finalConfidence) {
            finalFiltered.push_back(detEllipses[i]);
            finalScores.push_back(detEllipseScore[i]);
        }
    }
    
    detEllipses.swap(finalFiltered);
    detEllipseScore.swap(finalScores);
}

void FLED::AdaptiveThresholdEnhancement(const cv::Mat& edgeImg) {
    if (detEllipses.empty() || edgeImg.empty()) return;
    
    std::vector<double> continuityScores;
    double maxContinuity = 0.0;
    
    for (size_t i = 0; i < detEllipses.size(); i++) {
        double cont = calculateEdgeContinuity(edgeImg, detEllipses[i], 36);
        continuityScores.push_back(cont);
        if (cont > maxContinuity) {
            maxContinuity = cont;
        }
    }
    
    double threshold = maxContinuity * 0.4;
    
    std::vector<cv::RotatedRect> filtered;
    std::vector<double> filteredScores;
    
    for (size_t i = 0; i < detEllipses.size(); i++) {
        double combinedScore = detEllipseScore[i] * 0.5 + continuityScores[i] * 0.5;
        
        if (continuityScores[i] >= threshold && combinedScore > 0.25) {
            filtered.push_back(detEllipses[i]);
            filteredScores.push_back(combinedScore);
        }
    }
    
    detEllipses.swap(filtered);
    detEllipseScore.swap(filteredScores);
}