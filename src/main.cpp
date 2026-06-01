#include "FLED.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

namespace
{
struct DemoOptions
{
	fs::path inputPath = fs::path(AAMED_OPENCV_PROJECT_ROOT) / "data" / "images" / "002_0038.jpg";
	fs::path outputDir = fs::path(AAMED_OPENCV_PROJECT_ROOT) / "output";
	bool exportDebug = false;
	bool quiet = false;
	double T_val = 0.77;
	double kd_radius_mul = 1.0;
	int region_bypass = 0;
};

void printUsage()
{
	std::cout
		<< "Usage: aamed_demo [--input <image>] [--output-dir <dir>] [--T-val <0.0-1.0>] [--kd-mul <1.0-N>] [--region-bypass <0|1|2>] [--export-debug] [--quiet]\n"
		<< "Example:\n"
		<< "  aamed_demo --input data/images/002_0038.jpg --output-dir output --T-val 0.5 --export-debug\n";
}

bool parseArgs(int argc, char **argv, DemoOptions &options)
{
	for (int idx = 1; idx < argc; ++idx)
	{
		const std::string arg = argv[idx];
		if (arg == "--input" && idx + 1 < argc)
		{
			options.inputPath = argv[++idx];
		}
		else if (arg == "--output-dir" && idx + 1 < argc)
		{
			options.outputDir = argv[++idx];
		}
		else if (arg == "--T-val" && idx + 1 < argc)
		{
			options.T_val = std::stod(argv[++idx]);
		}
		else if (arg == "--kd-mul" && idx + 1 < argc)
		{
			options.kd_radius_mul = std::stod(argv[++idx]);
		}
		else if (arg == "--region-bypass" && idx + 1 < argc)
		{
			options.region_bypass = std::stoi(argv[++idx]);
		}
		else if (arg == "--export-debug")
		{
			options.exportDebug = true;
		}
		else if (arg == "--quiet")
		{
			options.quiet = true;
		}
		else if (arg == "--help" || arg == "-h")
		{
			printUsage();
			return false;
		}
		else
		{
			std::cerr << "Unknown argument: " << arg << '\n';
			printUsage();
			return false;
		}
	}
	return true;
}

double totalDetectionTimeMs(const cv::Vec<double, 10> &detailTime)
{
	return detailTime[0] + detailTime[1] + detailTime[2] + detailTime[3] + detailTime[6] + detailTime[9];
}

void drawDetectedEllipses(const std::vector<cv::RotatedRect> &ellipses, cv::Mat &canvas)
{
	for (const auto &ellipse_data : ellipses)
	{
		cv::RotatedRect drawEllipse;
		drawEllipse.center.x = ellipse_data.center.y;
		drawEllipse.center.y = ellipse_data.center.x;
		drawEllipse.size.height = ellipse_data.size.width;
		drawEllipse.size.width = ellipse_data.size.height;
		drawEllipse.angle = -ellipse_data.angle;
		cv::ellipse(canvas, drawEllipse, cv::Scalar(0, 0, 255), 2);
	}
}
}

int main(int argc, char **argv)
{
	DemoOptions options;
	if (!parseArgs(argc, argv, options))
	{
		return argc > 1 ? 1 : 0;
	}

	if (!fs::exists(options.inputPath))
	{
		std::cerr << "Input image not found: " << options.inputPath.string() << std::endl;
		return 1;
	}

	fs::create_directories(options.outputDir);

	cv::Mat imgColor = cv::imread(options.inputPath.string(), cv::IMREAD_COLOR);
	if (imgColor.empty())
	{
		std::cerr << "Failed to read image: " << options.inputPath.string() << std::endl;
		return 1;
	}

	cv::Mat imgGray;
	cv::cvtColor(imgColor, imgGray, cv::COLOR_BGR2GRAY);

	AAMED aamed(imgGray.rows + 16, imgGray.cols + 16);
	aamed.SetParameters(CV_PI / 3, 3.4, options.T_val);
	aamed.SetKDTotalRadiusMul(options.kd_radius_mul);
	aamed.SetRegionBypass(options.region_bypass);
	aamed.run_FLED(imgGray);

	cv::Vec<double, 10> detailTime = cv::Vec<double, 10>::all(0);
	aamed.showDetailBreakdown(detailTime, options.quiet ? 0 : 1);

	cv::Mat result = imgColor.clone();
	drawDetectedEllipses(aamed.detEllipses, result);

	const fs::path detectedImagePath = options.outputDir / "detected.png";
	const fs::path resultFilePath = options.outputDir / (options.inputPath.filename().string() + ".fled.txt");
	const fs::path timingPath = options.outputDir / "timing.txt";

	if (!cv::imwrite(detectedImagePath.string(), result))
	{
		std::cerr << "Failed to write result image: " << detectedImagePath.string() << std::endl;
		return 1;
	}

	aamed.writeFLED(options.outputDir.string() + "/", resultFilePath.filename().string(), totalDetectionTimeMs(detailTime));
	aamed.writeDetectionsTable((options.outputDir / "detections.txt").string());

	{
		std::ofstream timingOut(timingPath);
		timingOut << "PreProcessingMs " << detailTime[0] << '\n';
		timingOut << "ArcSegmentationMs " << detailTime[1] << '\n';
		timingOut << "ArcGroupingMs " << detailTime[2] << '\n';
		timingOut << "EllipseFittingMs " << detailTime[3] << '\n';
		timingOut << "EllipseValidationMs " << detailTime[6] << '\n';
		timingOut << "EllipseClusterMs " << detailTime[9] << '\n';
		timingOut << "TotalMs " << totalDetectionTimeMs(detailTime) << '\n';
	}

	if (options.exportDebug)
	{
		aamed.exportDebugArtifacts((options.outputDir / "debug").string(), imgGray, &detailTime);
	}

	std::cout << "Input: " << options.inputPath.string() << '\n';
	std::cout << "Detections: " << aamed.detEllipses.size() << '\n';
	std::cout << "Saved image: " << detectedImagePath.string() << '\n';
	std::cout << "Saved results: " << resultFilePath.string() << '\n';
	if (options.exportDebug)
	{
		std::cout << "Saved debug artifacts: " << (options.outputDir / "debug").string() << '\n';
	}
	return 0;
}
