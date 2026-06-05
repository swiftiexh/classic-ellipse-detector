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
	int scc_opt_level = 0;
	int scc_cluster_mode = 0;
	double scc_iou_threshold = 0.7;
	SCCDeferredConfig deferred;
	int selector_profile = 0;
	bool exportCandidates = false;
	fs::path inputListPath;
	double scale_factor = 1.0;
	int contour_mode = 0;
	double canny_scale_adj = 1.0;
	int scale_mode = 0;  // 0=INTER_CUBIC, 1=Laplacian-enhanced, 2=CLAHE+CUBIC
	int fsa_relax = 0;   // 0=standard FSA, 1=relaxed (theta*1.5, length*0.5)
};

void applySelectorProfile(int profile, SCCDeferredConfig &config)
{
	if (profile == 1)
	{
		config.weightNewSupport = 0;
		config.weightDuplicate = 0;
		config.weightNegative = 0;
		config.weightSharedArc = 0;
		config.minGain = 0.55;
		return;
	}
	if (profile >= 2 && profile <= 10)
	{
		const int index = profile - 2;
		const double supportValues[3] = { 0.15, 0.30, 0.45 };
		const double duplicateValues[3] = { 0.25, 0.50, 0.75 };
		config.weightNewSupport = supportValues[index / 3];
		config.weightDuplicate = duplicateValues[index % 3];
		return;
	}
	switch (profile)
	{
	case 11: config.weightNegative = 0.075; break;
	case 12: config.weightNegative = 0.30; break;
	case 13: config.weightSharedArc = 0.05; break;
	case 14: config.weightSharedArc = 0.20; break;
	case 15: config.minGain = 0.45; break;
	case 16: config.minGain = 0.65; break;
	default: break;
	}
}

void printUsage()
{
	std::cout
		<< "Usage: aamed_demo [--input <image>] [--input-list <file>] [--output-dir <dir>]\n"
			<< "       [--T-val <0.0-1.0>] [--kd-mul <1.0-N>] [--region-bypass <0|1|2>]\n"
			<< "       [--scale-factor <1.0-N>] [--contour-mode <0|1>] [--canny-scale-adj <float>]\n"
			<< "       [--scc-opt-level <0|1|2|3>] [--scc-cluster-mode <0|1>]\n"
			<< "       [--scc-iou-threshold <0.0-1.0>] [--scc-selection-mode <0|1|2|3>]\n"
			<< "       [--scc-proposal-T-val <0.0-1.0>] [--scc-top-k <int>]\n"
			<< "       [--scc-max-tested-per-root <int>] [--scc-max-candidates <int>]\n"
			<< "       [--scc-selector-profile <0-16>] [--scc-weight-new-support <float>]\n"
			<< "       [--scc-weight-duplicate <float>] [--scc-weight-negative <float>]\n"
			<< "       [--scc-weight-shared-arc <float>] [--scc-min-gain <float>]\n"
			<< "       [--scc-rescue <0|1>] [--scc-rescue-max <int>] [--scc-rescue-mad <float>]\n"
			<< "       [--scc-export-candidates] [--export-debug] [--quiet]\n"
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
		else if (arg == "--scc-opt-level" && idx + 1 < argc)
		{
			options.scc_opt_level = std::stoi(argv[++idx]);
		}
		else if (arg == "--scc-cluster-mode" && idx + 1 < argc)
		{
			options.scc_cluster_mode = std::stoi(argv[++idx]);
		}
		else if (arg == "--scc-iou-threshold" && idx + 1 < argc)
		{
			options.scc_iou_threshold = std::stod(argv[++idx]);
		}
		else if (arg == "--scc-selection-mode" && idx + 1 < argc)
		{
			options.deferred.mode = std::stoi(argv[++idx]);
		}
		else if (arg == "--scc-proposal-T-val" && idx + 1 < argc)
		{
			options.deferred.proposalThreshold = std::stod(argv[++idx]);
		}
		else if (arg == "--scc-top-k" && idx + 1 < argc)
		{
			options.deferred.topKPerRoot = std::stoi(argv[++idx]);
		}
		else if (arg == "--scc-max-tested-per-root" && idx + 1 < argc)
		{
			options.deferred.maxTestedPerRoot = std::stoi(argv[++idx]);
		}
		else if (arg == "--scc-max-candidates" && idx + 1 < argc)
		{
			options.deferred.maxCandidates = std::stoi(argv[++idx]);
		}
		else if (arg == "--scc-selector-profile" && idx + 1 < argc)
		{
			options.selector_profile = std::stoi(argv[++idx]);
		}
		else if (arg == "--scc-weight-new-support" && idx + 1 < argc)
		{
			options.deferred.weightNewSupport = std::stod(argv[++idx]);
		}
		else if (arg == "--scc-weight-duplicate" && idx + 1 < argc)
		{
			options.deferred.weightDuplicate = std::stod(argv[++idx]);
		}
		else if (arg == "--scc-weight-negative" && idx + 1 < argc)
		{
			options.deferred.weightNegative = std::stod(argv[++idx]);
		}
		else if (arg == "--scc-weight-shared-arc" && idx + 1 < argc)
		{
			options.deferred.weightSharedArc = std::stod(argv[++idx]);
		}
		else if (arg == "--scc-min-gain" && idx + 1 < argc)
		{
			options.deferred.minGain = std::stod(argv[++idx]);
		}
		else if (arg == "--scc-rescue" && idx + 1 < argc)
		{
			options.deferred.rescueEnabled = std::stoi(argv[++idx]) != 0;
		}
		else if (arg == "--scc-rescue-max" && idx + 1 < argc)
		{
			options.deferred.rescueMaxAttempts = std::stoi(argv[++idx]);
		}
		else if (arg == "--scc-rescue-mad" && idx + 1 < argc)
		{
			options.deferred.rescueMadMultiplier = std::stod(argv[++idx]);
		}
		else if (arg == "--scc-export-candidates")
		{
			options.exportCandidates = true;
		}
		else if (arg == "--export-debug")
		{
			options.exportDebug = true;
		}
		else if (arg == "--quiet")
		{
			options.quiet = true;
		}
		else if (arg == "--input-list" && idx + 1 < argc)
		{
			options.inputListPath = argv[++idx];
		}
		else if (arg == "--scale-factor" && idx + 1 < argc)
		{
			options.scale_factor = std::stod(argv[++idx]);
		}
		else if (arg == "--contour-mode" && idx + 1 < argc)
		{
			options.contour_mode = std::stoi(argv[++idx]);
		}
		else if (arg == "--canny-scale-adj" && idx + 1 < argc)
		{
			options.canny_scale_adj = std::stod(argv[++idx]);
		}
		else if (arg == "--scale-mode" && idx + 1 < argc)
		{
			options.scale_mode = std::stoi(argv[++idx]);
		}
		else if (arg == "--fsa-relax" && idx + 1 < argc)
		{
			options.fsa_relax = std::stoi(argv[++idx]);
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

/// Process a single image through AAMED and write results.
/// Returns the number of detected ellipses, or -1 on error.
int processImage(const fs::path &imagePath, const fs::path &outputDir, DemoOptions options)
{
	if (!fs::exists(imagePath))
	{
		std::cerr << "Input image not found: " << imagePath.string() << std::endl;
		return -1;
	}

	cv::Mat imgColor = cv::imread(imagePath.string(), cv::IMREAD_COLOR);
	if (imgColor.empty())
	{
		std::cerr << "Failed to read image: " << imagePath.string() << std::endl;
		return -1;
	}

	cv::Mat imgGray;
	cv::cvtColor(imgColor, imgGray, cv::COLOR_BGR2GRAY);

	// Internal upscaling (JIT, no temp file needed)
	int scaledRows = imgGray.rows, scaledCols = imgGray.cols;
	cv::Mat imgScaled;
	if (std::abs(options.scale_factor - 1.0) > 1e-6)
	{
		scaledRows = static_cast<int>(imgGray.rows * options.scale_factor);
		scaledCols = static_cast<int>(imgGray.cols * options.scale_factor);
		cv::Mat imgEnhanced;
		if (options.scale_mode == 1)
		{
			// Laplacian edge enhancement before upscaling
			// img_enhanced = img * 1.5 - GaussianBlur(img, 0.5)
			cv::Mat blurred;
			cv::GaussianBlur(imgGray, blurred, cv::Size(0, 0), 0.5);
			cv::addWeighted(imgGray, 1.5, blurred, -1.0, 0, imgEnhanced);
		}
		else if (options.scale_mode == 2)
		{
			// CLAHE contrast enhancement then upscale
			cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
			clahe->apply(imgGray, imgEnhanced);
		}
		else
		{
			imgEnhanced = imgGray;
		}
		cv::resize(imgEnhanced, imgScaled, cv::Size(scaledCols, scaledRows), 0, 0, cv::INTER_CUBIC);
	}
	else
	{
		imgScaled = imgGray;
	}

	AAMED aamed(scaledRows + 16, scaledCols + 16);
	double theta_fsa = CV_PI / 3;
	double length_fsa = 3.4;
	if (options.fsa_relax >= 1)
	{
		// Relaxed FSA: wider angle window, shorter allowed arc length
		theta_fsa *= 1.5;   // PI/3 * 1.5 = PI/2
		length_fsa *= 0.5;  // 3.4 * 0.5 = 1.7
	}
	aamed.SetParameters(theta_fsa, length_fsa, options.T_val);
	aamed.SetKDTotalRadiusMul(options.kd_radius_mul);
	aamed.SetRegionBypass(options.region_bypass);
	aamed.SetSCCOptLevel(options.scc_opt_level);
	aamed.SetSCCClusterMode(options.scc_cluster_mode);
	aamed.SetSCCIOUThreshold(options.scc_iou_threshold);
	applySelectorProfile(options.selector_profile, options.deferred);
	aamed.SetSCCDeferredConfig(options.deferred);
	aamed.SetSCCExportCandidates(options.exportCandidates);
	aamed.run_FLED(imgScaled);

	cv::Vec<double, 10> detailTime = cv::Vec<double, 10>::all(0);
	aamed.showDetailBreakdown(detailTime, options.quiet ? 0 : 1);

	const double totalMs = totalDetectionTimeMs(detailTime);

	// Write result in fled.txt format with remapped coordinates if scaled
	const std::string resultFileName = imagePath.filename().string() + ".fled.txt";
	std::ofstream fledOut(outputDir / resultFileName);
	if (!fledOut)
	{
		std::cerr << "Failed to write: " << (outputDir / resultFileName).string() << std::endl;
		return -1;
	}
	fledOut << totalMs << '\n';
	for (const auto &e : aamed.detEllipses)
	{
		// detEllipses format: center(y,x), size(height,width), angle(degrees)
		// fled.txt format: type row col height width angle_deg
		double cx = e.center.y / options.scale_factor;
		double cy = e.center.x / options.scale_factor;
		double a  = e.size.height / options.scale_factor;
		double b  = e.size.width  / options.scale_factor;
		double angle = e.angle;
		fledOut << "1 " << cy << ' ' << cx << ' ' << a << ' ' << b << ' ' << angle << '\n';
	}
	fledOut.close();

	if (!options.quiet && !options.inputListPath.empty())
	{
		std::cout << imagePath.filename().string() << ": " << aamed.detEllipses.size()
		          << " detections, " << totalMs << " ms" << std::endl;
	}

	if (options.exportCandidates)
	{
		const std::string imageName = imagePath.filename().string();
		aamed.writeSCCCandidateDiagnostics(
			(outputDir / (imageName + ".candidates.tsv")).string(),
			(outputDir / (imageName + ".selection.tsv")).string());
	}

	if (options.exportDebug)
	{
		aamed.exportDebugArtifacts((outputDir / "debug").string(), imgScaled, &detailTime);
	}

	return static_cast<int>(aamed.detEllipses.size());
}

int main(int argc, char **argv)
{
	DemoOptions options;
	if (!parseArgs(argc, argv, options))
	{
		return argc > 1 ? 1 : 0;
	}

	fs::create_directories(options.outputDir);

	// Batch mode: process all images listed in --input-list
	if (!options.inputListPath.empty())
	{
		if (!fs::exists(options.inputListPath))
		{
			std::cerr << "Input list not found: " << options.inputListPath.string() << std::endl;
			return 1;
		}
		std::ifstream listFile(options.inputListPath.string());
		if (!listFile)
		{
			std::cerr << "Cannot open input list: " << options.inputListPath.string() << std::endl;
			return 1;
		}
		std::vector<fs::path> imagePaths;
		std::string line;
		while (std::getline(listFile, line))
		{
			if (!line.empty() && line[0] != '#')
				imagePaths.emplace_back(line);
		}
		listFile.close();

		if (!options.quiet)
		{
			std::cout << "Batch mode: " << imagePaths.size() << " images, scale="
			          << options.scale_factor << "x, T=" << options.T_val << std::endl;
		}

		int ok = 0, fail = 0;
		for (const auto &imgPath : imagePaths)
		{
			int result = processImage(imgPath, options.outputDir, options);
			if (result >= 0) ok++; else fail++;
		}

		if (!options.quiet)
			std::cout << "Batch done: ok=" << ok << " fail=" << fail << std::endl;
		return fail > 0 ? 1 : 0;
	}

	// Single-image mode
	if (!fs::exists(options.inputPath))
	{
		std::cerr << "Input image not found: " << options.inputPath.string() << std::endl;
		return 1;
	}

	fs::create_directories(options.outputDir);

	int detCount = processImage(options.inputPath, options.outputDir, options);
	if (detCount < 0) return 1;

	if (!options.quiet)
	{
		std::cout << "Input: " << options.inputPath.string() << '\n';
		std::cout << "Detections: " << detCount << '\n';
		if (options.exportDebug)
			std::cout << "Saved debug artifacts: " << (options.outputDir / "debug").string() << '\n';
	}
	return 0;
}
