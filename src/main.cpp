#include "ExperimentConfig.h"
#include "FLED.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace
{
std::vector<std::string> readImageNames(const fs::path &path)
{
	if (!path.empty() && fs::exists(path))
	{
	std::ifstream in(path);
	if (!in)
	{
		throw std::runtime_error("Failed to open image list: " + path.string());
	}

	std::vector<std::string> names;
	std::string line;
	while (std::getline(in, line))
	{
		if (!line.empty())
		{
			names.push_back(line);
		}
	}
	return names;
	}

	return {};
}

std::vector<std::string> enumerateImageNames(const fs::path &imagesDir)
{
	if (!fs::exists(imagesDir))
	{
		throw std::runtime_error("Images directory does not exist: " + imagesDir.string());
	}

	std::vector<std::string> names;
	for (const auto &entry : fs::directory_iterator(imagesDir))
	{
		if (!entry.is_regular_file())
		{
			continue;
		}

		const std::string extension = entry.path().extension().string();
		if (extension == ".jpg" || extension == ".png" || extension == ".bmp" ||
			extension == ".jpeg" || extension == ".tif" || extension == ".tiff")
		{
			names.push_back(entry.path().filename().string());
		}
	}

	std::sort(names.begin(), names.end());
	return names;
}

fs::path resolveImagePath(const fs::path &imagesDir, const std::string &imageName)
{
	const fs::path directPath = imagesDir / imageName;
	if (fs::exists(directPath))
	{
		return directPath;
	}

	const fs::path namePath(imageName);
	if (namePath.has_extension())
	{
		return directPath;
	}

	static const char *kExtensions[] = {".jpg", ".png", ".bmp", ".jpeg", ".tif", ".tiff"};
	for (const char *extension : kExtensions)
	{
		const fs::path candidate = imagesDir / (imageName + extension);
		if (fs::exists(candidate))
		{
			return candidate;
		}
	}

	return directPath;
}

double detailBreakdownTotalMs(const cv::Vec<double, 10> &detailTime)
{
	return detailTime[0] + detailTime[1] + detailTime[2] + detailTime[3] + detailTime[6] + detailTime[9];
}

void drawDetectedEllipses(const std::vector<cv::RotatedRect> &ellipses, cv::Mat &canvas)
{
	for (const auto &ellipseData : ellipses)
	{
		cv::RotatedRect drawEllipse;
		drawEllipse.center.x = ellipseData.center.y;
		drawEllipse.center.y = ellipseData.center.x;
		drawEllipse.size.height = ellipseData.size.width;
		drawEllipse.size.width = ellipseData.size.height;
		drawEllipse.angle = -ellipseData.angle;
		cv::ellipse(canvas, drawEllipse, cv::Scalar(0, 0, 255), 2);
	}
}

struct DetectionRunResult
{
	bool success = false;
	size_t detections = 0;
	double totalMs = 0.0;
	double detailBreakdownMs = 0.0;
	fs::path resultFilePath;
};

DetectionRunResult runDetection(
	const fs::path &inputPath,
	const fs::path &outputDir,
	const experiment::ExperimentConfig &config,
	bool exportDebug,
	bool quiet,
	bool writeVisualization)
{
	DetectionRunResult result;

	if (!fs::exists(inputPath))
	{
		std::cerr << "Input image not found: " << inputPath.string() << '\n';
		return result;
	}

	fs::create_directories(outputDir);

	cv::Mat imgColor = cv::imread(inputPath.string(), cv::IMREAD_COLOR);
	if (imgColor.empty())
	{
		std::cerr << "Failed to read image: " << inputPath.string() << '\n';
		return result;
	}

	cv::Mat imgGray;
	cv::cvtColor(imgColor, imgGray, cv::COLOR_BGR2GRAY);

	int maxScale = 1;
	if (config.multiScaleFpn.enable)
	{
		for (int scale : config.multiScaleFpn.scales)
		{
			maxScale = std::max(maxScale, scale);
		}
	}

	AAMED aamed(imgGray.rows * maxScale + 16, imgGray.cols * maxScale + 16);
	aamed.SetParameters(config.detector.thetaArc, config.detector.lambdaArc, config.detector.validationThreshold);
	aamed.SetWeightedArcConfig(config.weightedArc);
	aamed.SetMultiScaleConfig(config.multiScaleFpn);
	aamed.SetSmallEllipseGuardConfig(config.smallEllipseGuard);

	const double tic = cv::getTickCount();
	if (config.multiScaleFpn.enable)
	{
		aamed.run_FLED_MultiScale(imgGray);
	}
	else
	{
		aamed.run_FLED(imgGray);
	}
	result.totalMs = (cv::getTickCount() - tic) * 1000.0 / cv::getTickFrequency();

	cv::Vec<double, 10> detailTime = cv::Vec<double, 10>::all(0);
	aamed.showDetailBreakdown(detailTime, quiet ? 0 : 1);
	result.detailBreakdownMs = detailBreakdownTotalMs(detailTime);
	result.detections = aamed.detEllipses.size();
	result.resultFilePath = outputDir / (inputPath.filename().string() + ".fled.txt");

	if (writeVisualization)
	{
		cv::Mat drawImage = imgColor.clone();
		drawDetectedEllipses(aamed.detEllipses, drawImage);

		const fs::path detectedImagePath = outputDir / "detected.png";
		if (!cv::imwrite(detectedImagePath.string(), drawImage))
		{
			std::cerr << "Failed to write result image: " << detectedImagePath.string() << '\n';
			return result;
		}

		std::ofstream timingOut(outputDir / "timing.txt");
		timingOut << "PreProcessingMs " << detailTime[0] << '\n';
		timingOut << "ArcSegmentationMs " << detailTime[1] << '\n';
		timingOut << "ArcGroupingMs " << detailTime[2] << '\n';
		timingOut << "EllipseFittingMs " << detailTime[3] << '\n';
		timingOut << "EllipseValidationMs " << detailTime[6] << '\n';
		timingOut << "EllipseClusterMs " << detailTime[9] << '\n';
		timingOut << "DetailBreakdownTotalMs " << result.detailBreakdownMs << '\n';
		timingOut << "WallClockTotalMs " << result.totalMs << '\n';

		aamed.writeDetectionsTable((outputDir / "detections.txt").string());
	}

	aamed.writeFLED(outputDir.string() + "/", result.resultFilePath.filename().string(), result.totalMs);

	if (exportDebug)
	{
		aamed.exportDebugArtifacts((outputDir / "debug").string(), imgGray, &detailTime);
	}

	result.success = true;
	return result;
}

int runSingleImageMode(const experiment::ExperimentConfig &config)
{
	const DetectionRunResult result = runDetection(
		config.single.inputPath,
		config.single.outputDir,
		config,
		config.single.exportDebug,
		config.single.quiet,
		true);

	if (!result.success)
	{
		return 1;
	}

	std::cout << "Mode: SingleImage\n";
	std::cout << "Experiment: " << config.experimentLabel << '\n';
	std::cout << "Input: " << config.single.inputPath.string() << '\n';
	std::cout << "Detections: " << result.detections << '\n';
	std::cout << "TotalMs: " << result.totalMs << '\n';
	std::cout << "Saved results: " << result.resultFilePath.string() << '\n';
	if (config.single.exportDebug)
	{
		std::cout << "Saved debug artifacts: " << (config.single.outputDir / "debug").string() << '\n';
	}
	return 0;
}

int runDatasetBatchMode(const experiment::ExperimentConfig &config)
{
	std::vector<std::string> imageNames = readImageNames(config.dataset.imageNamesPath);
	if (imageNames.empty())
	{
		imageNames = enumerateImageNames(config.dataset.imagesDir);
	}
	fs::create_directories(config.dataset.resultsDir);

	std::ofstream summaryOut(config.dataset.resultsDir / "batch_summary.txt");
	summaryOut << "ImageName Detections TotalMs Status\n";

	size_t successCount = 0;
	size_t totalDetections = 0;
	double totalMs = 0.0;

	for (const std::string &imageName : imageNames)
	{
		const fs::path inputPath = resolveImagePath(config.dataset.imagesDir, imageName);
		const DetectionRunResult result = runDetection(
			inputPath,
			config.dataset.resultsDir,
			config,
			config.dataset.exportDebug,
			config.dataset.quiet,
			false);

		if (!result.success)
		{
			summaryOut << imageName << " 0 0 FAIL\n";
			std::cerr << "Failed: " << imageName << '\n';
			continue;
		}

		++successCount;
		totalDetections += result.detections;
		totalMs += result.totalMs;

		summaryOut << inputPath.filename().string() << ' '
			<< result.detections << ' '
			<< result.totalMs << ' '
			<< "OK\n";
	}

	const double averageMs = successCount == 0 ? 0.0 : totalMs / static_cast<double>(successCount);
	summaryOut << "Summary " << successCount << ' ' << totalDetections << ' ' << averageMs << " AVG_MS\n";

	std::cout << "Mode: DatasetBatch\n";
	std::cout << "Experiment: " << config.experimentLabel << '\n';
	std::cout << "DatasetRoot: " << config.dataset.datasetRoot.string() << '\n';
	std::cout << "ProcessedImages: " << successCount << "/" << imageNames.size() << '\n';
	std::cout << "TotalDetections: " << totalDetections << '\n';
	std::cout << "AverageMs: " << averageMs << '\n';
	std::cout << "ResultsDir: " << config.dataset.resultsDir.string() << '\n';
	return successCount == imageNames.size() ? 0 : 1;
}
}

int main()
{
	try
	{
		const experiment::ExperimentConfig config = experiment::BuildExperimentConfig();
		if (config.mode == experiment::RunMode::SingleImage)
		{
			return runSingleImageMode(config);
		}
		return runDatasetBatchMode(config);
	}
	catch (const std::exception &ex)
	{
		std::cerr << "Run failed: " << ex.what() << '\n';
		return 1;
	}
}
