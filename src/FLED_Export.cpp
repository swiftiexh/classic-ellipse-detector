#include "FLED.h"

#include <cstdio>
#include <filesystem>
#include <sstream>

namespace fs = std::filesystem;

namespace
{
std::string ensureDirectoryString(const fs::path &dir)
{
	std::string value = dir.string();
	if (!value.empty() && value.back() != '/' && value.back() != '\\')
	{
		value.push_back('/');
	}
	return value;
}

cv::Mat createEdgeMapPreview(const cv::Mat &edgeMap)
{
	cv::Mat preview = cv::Mat::zeros(edgeMap.size(), CV_8UC1);
	for (int row = 0; row < edgeMap.rows; ++row)
	{
		for (int col = 0; col < edgeMap.cols; ++col)
		{
			preview.at<unsigned char>(row, col) = edgeMap.at<unsigned char>(row, col) == 0 ? 0 : 255;
		}
	}
	return preview;
}

cv::Mat renderPolylineContours(
	const std::vector<std::vector<cv::Point>> &contours,
	int rows,
	int cols,
	const cv::Scalar &lineColor,
	bool drawIndices)
{
	cv::Mat canvas(rows, cols, CV_8UC3, cv::Scalar(255, 255, 255));
	char label[32];
	for (int contourIdx = 0; contourIdx < static_cast<int>(contours.size()); ++contourIdx)
	{
		const auto &contour = contours[contourIdx];
		if (contour.empty())
		{
			continue;
		}

		if (contour.size() == 1)
		{
			cv::circle(canvas, cv::Point(contour[0].y, contour[0].x), 1, lineColor, 1);
		}
		else
		{
			for (int pointIdx = 0; pointIdx + 1 < static_cast<int>(contour.size()); ++pointIdx)
			{
				cv::line(
					canvas,
					cv::Point(contour[pointIdx].y, contour[pointIdx].x),
					cv::Point(contour[pointIdx + 1].y, contour[pointIdx + 1].x),
					lineColor,
					1);
			}
		}

		cv::circle(canvas, cv::Point(contour.front().y, contour.front().x), 2, cv::Scalar(0, 0, 255), -1);
		if (drawIndices)
		{
#if defined(__GNUC__)
			std::sprintf(label, "%d", contourIdx);
#else
			sprintf_s(label, "%d", contourIdx);
#endif
			const cv::Point anchor = contour[contour.size() / 2];
			cv::putText(canvas, label, cv::Point(anchor.y, anchor.x + 2), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);
		}
	}
	return canvas;
}

void writeContourCollection(const fs::path &filepath, const std::vector<std::vector<cv::Point>> &contours)
{
	std::ofstream out(filepath);
	out << "# contour_count " << contours.size() << '\n';
	for (size_t contourIdx = 0; contourIdx < contours.size(); ++contourIdx)
	{
		out << "contour " << contourIdx << " size " << contours[contourIdx].size() << '\n';
		for (const auto &point : contours[contourIdx])
		{
			out << point.x << ' ' << point.y << '\n';
		}
	}
}

void writeTimingBreakdown(const fs::path &filepath, const cv::Vec<double, 10> &timing)
{
	const double total = timing[0] + timing[1] + timing[2] + timing[3] + timing[6] + timing[9];
	std::ofstream out(filepath);
	out << "PreProcessingMs " << timing[0] << '\n';
	out << "ArcSegmentationMs " << timing[1] << '\n';
	out << "ArcGroupingMs " << timing[2] << '\n';
	out << "EllipseFittingMs " << timing[3] << '\n';
	out << "EllipseFittingCount " << timing[4] << '\n';
	out << "EllipseFittingAvgMs " << timing[5] << '\n';
	out << "EllipseValidationMs " << timing[6] << '\n';
	out << "EllipseValidationCount " << timing[7] << '\n';
	out << "EllipseValidationAvgMs " << timing[8] << '\n';
	out << "EllipseClusterMs " << timing[9] << '\n';
	out << "TotalMs " << total << '\n';
}

cv::Mat renderAdjacencyMatrix(const char *linkMatrix, int arcCount)
{
	const int safeArcCount = std::max(arcCount, 1);
	cv::Mat matrix(safeArcCount, safeArcCount, CV_8UC1, cv::Scalar(127));
	for (int row = 0; row < arcCount; ++row)
	{
		for (int col = 0; col < arcCount; ++col)
		{
			const char value = linkMatrix[row * arcCount + col];
			unsigned char pixel = 127;
			if (value > 0)
			{
				pixel = 255;
			}
			else if (value < 0)
			{
				pixel = 0;
			}
			matrix.at<unsigned char>(row, col) = pixel;
		}
	}
	return matrix;
}

void drawDetectedEllipses(
	const std::vector<cv::RotatedRect> &ellipses,
	const std::vector<double> &scores,
	cv::Mat &canvas)
{
	char scoreText[32];
	for (size_t idx = 0; idx < ellipses.size(); ++idx)
	{
		cv::RotatedRect drawEllipse;
		drawEllipse.center.x = ellipses[idx].center.y;
		drawEllipse.center.y = ellipses[idx].center.x;
		drawEllipse.size.height = ellipses[idx].size.width;
		drawEllipse.size.width = ellipses[idx].size.height;
		drawEllipse.angle = -ellipses[idx].angle;
		cv::ellipse(canvas, drawEllipse, cv::Scalar(0, 0, 255), 2);
#if defined(__GNUC__)
		std::sprintf(scoreText, "%.2f", scores[idx]);
#else
		sprintf_s(scoreText, "%.2f", scores[idx]);
#endif
		cv::putText(canvas, scoreText, drawEllipse.center, cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 255), 1);
	}
}
}

void FLED::writeDetectionsTable(const string &filepath)
{
	std::ofstream out(filepath);
	out << "# center_row center_col major_diameter minor_diameter angle_deg score\n";
	for (size_t idx = 0; idx < detEllipses.size(); ++idx)
	{
		out << detEllipses[idx].center.x << ' '
			<< detEllipses[idx].center.y << ' '
			<< detEllipses[idx].size.width << ' '
			<< detEllipses[idx].size.height << ' '
			<< detEllipses[idx].angle << ' '
			<< detEllipseScore[idx] << '\n';
	}
}

void FLED::exportDebugArtifacts(const string &outputDir, const Mat &sourceGray, const cv::Vec<double, 10> *detDetailTime)
{
	const fs::path exportDir(outputDir);
	fs::create_directories(exportDir);

	if (!imgCanny.empty())
	{
		cv::imwrite((exportDir / "01_edge_map.png").string(), createEdgeMapPreview(imgCanny));
	}

	cv::imwrite((exportDir / "02_edge_contours.png").string(), renderPolylineContours(edgeContours, iROWS, iCOLS, cv::Scalar(0, 0, 0), false));
	writeContourCollection(exportDir / "02_edge_contours.txt", edgeContours);

	cv::imwrite((exportDir / "03_dp_contours.png").string(), renderPolylineContours(dpContours, iROWS, iCOLS, cv::Scalar(255, 0, 0), true));
	writeContourCollection(exportDir / "03_dp_contours.txt", dpContours);

	cv::imwrite((exportDir / "04_fsa_arcs.png").string(), renderPolylineContours(FSA_ArcContours, iROWS, iCOLS, cv::Scalar(255, 0, 0), true));
	writeContourCollection(exportDir / "04_fsa_arcs.txt", FSA_ArcContours);

	const int arcCount = static_cast<int>(FSA_ArcContours.size());
	{
		std::ofstream matrixOut(exportDir / "05_adjacency_matrix.csv");
		const char *linkMatrix = LinkMatrix.GetDataPoint();
		for (int row = 0; row < arcCount; ++row)
		{
			for (int col = 0; col < arcCount; ++col)
			{
				if (col > 0)
				{
					matrixOut << ',';
				}
				matrixOut << static_cast<int>(linkMatrix[row * arcCount + col]);
			}
			matrixOut << '\n';
		}
		cv::imwrite((exportDir / "05_adjacency_matrix.png").string(), renderAdjacencyMatrix(linkMatrix, arcCount));
	}

	cv::Mat ellipseCanvas;
	if (sourceGray.empty())
	{
		ellipseCanvas = cv::Mat::zeros(iROWS, iCOLS, CV_8UC3);
	}
	else if (sourceGray.channels() == 1)
	{
		cv::cvtColor(sourceGray.clone(), ellipseCanvas, cv::COLOR_GRAY2BGR);
	}
	else
	{
		ellipseCanvas = sourceGray.clone();
	}
	drawDetectedEllipses(detEllipses, detEllipseScore, ellipseCanvas);
	cv::imwrite((exportDir / "06_detected_ellipses.png").string(), ellipseCanvas);

	double totalTime = 0.0;
	if (detDetailTime != nullptr)
	{
		totalTime = (*detDetailTime)[0] + (*detDetailTime)[1] + (*detDetailTime)[2] + (*detDetailTime)[3] + (*detDetailTime)[6] + (*detDetailTime)[9];
		writeTimingBreakdown(exportDir / "07_timing.txt", *detDetailTime);
	}

	writeFLED(ensureDirectoryString(exportDir), "06_detections.fled.txt", totalTime);
	writeDetectionsTable((exportDir / "06_detections.txt").string());

	std::ofstream summaryOut(exportDir / "08_summary.txt");
	summaryOut << "EdgeContours " << edgeContours.size() << '\n';
	summaryOut << "DPContours " << dpContours.size() << '\n';
	summaryOut << "FSAArcs " << FSA_ArcContours.size() << '\n';
	summaryOut << "Detections " << detEllipses.size() << '\n';
}
