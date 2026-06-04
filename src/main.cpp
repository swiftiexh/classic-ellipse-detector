#include "FLED.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace
{
enum class RunMode
{
	SingleImage,
	DatasetBatch
};

struct DetectorParams
{
	double thetaArc = CV_PI / 3.0;
	double lambdaArc = 3.4;
	double validationThreshold = 0.77;
};

struct SingleImageConfig
{
	fs::path inputPath;
	fs::path outputDir;
	bool exportDebug = true;
	bool quiet = false;
};

struct DatasetBatchConfig
{
	fs::path datasetRoot;
	fs::path imagesDir;
	fs::path imageNamesPath;
	fs::path resultsDir;
	bool exportDebug = false;
	bool quiet = true;
};

struct AppConfig
{
	RunMode mode = RunMode::SingleImage;
	DetectorParams detector;
	SingleImageConfig single;
	DatasetBatchConfig batch;
};

AppConfig buildConfig()
{
	const fs::path projectRoot(AAMED_OPENCV_PROJECT_ROOT);

	AppConfig config;
	config.mode = RunMode::DatasetBatch;
	config.single.inputPath = projectRoot / "demo" / "002_0038.jpg";
	config.single.outputDir = projectRoot / "output" / "single";

	config.batch.datasetRoot = projectRoot / "dataset";
	config.batch.imagesDir = config.batch.datasetRoot / "images";
	config.batch.imageNamesPath = config.batch.datasetRoot / "imagenames.txt";
	config.batch.resultsDir = projectRoot / "output" / "dataset";

	return config;
}

std::vector<std::string> readImageNames(const fs::path &path)
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

double totalDetectionTimeMs(const cv::Vec<double, 10> &detailTime)
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
	fs::path resultFilePath;
};

DetectionRunResult runDetection(
	const fs::path &inputPath,
	const fs::path &outputDir,
	const DetectorParams &detector,
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

	AAMED aamed(imgGray.rows + 16, imgGray.cols + 16);
	aamed.SetParameters(detector.thetaArc, detector.lambdaArc, detector.validationThreshold);
	aamed.run_FLED(imgGray);

	cv::Vec<double, 10> detailTime = cv::Vec<double, 10>::all(0);
	aamed.showDetailBreakdown(detailTime, quiet ? 0 : 1);

	result.totalMs = totalDetectionTimeMs(detailTime);
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
		timingOut << "TotalMs " << result.totalMs << '\n';

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

int runSingleImageMode(const AppConfig &config)
{
	const DetectionRunResult result = runDetection(
		config.single.inputPath,
		config.single.outputDir,
		config.detector,
		config.single.exportDebug,
		config.single.quiet,
		true);

	if (!result.success)
	{
		return 1;
	}

	std::cout << "Mode: SingleImage\n";
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

int runDatasetBatchMode(const AppConfig &config)
{
	const std::vector<std::string> imageNames = readImageNames(config.batch.imageNamesPath);
	fs::create_directories(config.batch.resultsDir);

	std::ofstream summaryOut(config.batch.resultsDir / "batch_summary.txt");
	summaryOut << "ImageName Detections TotalMs Status\n";

	size_t successCount = 0;
	size_t totalDetections = 0;
	double totalMs = 0.0;

	for (const std::string &imageName : imageNames)
	{
		const fs::path inputPath = resolveImagePath(config.batch.imagesDir, imageName);
		const DetectionRunResult result = runDetection(
			inputPath,
			config.batch.resultsDir,
			config.detector,
			config.batch.exportDebug,
			config.batch.quiet,
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
	std::cout << "DatasetRoot: " << config.batch.datasetRoot.string() << '\n';
	std::cout << "ProcessedImages: " << successCount << "/" << imageNames.size() << '\n';
	std::cout << "TotalDetections: " << totalDetections << '\n';
	std::cout << "AverageMs: " << averageMs << '\n';
	std::cout << "ResultsDir: " << config.batch.resultsDir.string() << '\n';
	return successCount == imageNames.size() ? 0 : 1;
}
}

int main()
{
	try
	{
		const AppConfig config = buildConfig();
		if (config.mode == RunMode::SingleImage)
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
