#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace
{
constexpr double kPi = 3.14159265358979323846;

struct EllipseParam
{
	double cx = 0.0;
	double cy = 0.0;
	double a = 0.0;
	double b = 0.0;
	double thetaRad = 0.0;
	double score = 0.0;
};

struct DetectionFile
{
	double timeMs = 0.0;
	std::vector<EllipseParam> ellipses;
};

enum class GroundTruthFormat
{
	PlainRad,
	PlainDeg,
	RandomGt,
	PrasadGt,
	ConcentricGt
};

enum class ResultFormat
{
	AamedFled,
	PlainRad,
	PlainDeg
};

struct EvalOptions
{
	std::optional<fs::path> datasetRoot;
	std::optional<fs::path> imageNamesPath;
	std::optional<fs::path> gtDir;
	std::optional<fs::path> resultsDir;
	std::optional<fs::path> reportPath;
	GroundTruthFormat gtFormat = GroundTruthFormat::PlainRad;
	ResultFormat resultFormat = ResultFormat::AamedFled;
	std::string gtPrefix;
	std::string gtSuffix = ".txt";
	std::string resultSuffix = ".fled.txt";
	double overlapThreshold = 0.8;
};

void printUsage()
{
	std::cout
		<< "Usage: aamed_eval --results-dir <dir> [--dataset-root <dir> | --imagenames <file> --gt-dir <dir>]\n"
		<< "                  [--gt-format plain_rad|plain_deg|random|prasad|concentric]\n"
		<< "                  [--result-format aamed_fled|plain_rad|plain_deg]\n"
		<< "                  [--gt-prefix <prefix>] [--gt-suffix <suffix>] [--result-suffix <suffix>]\n"
		<< "                  [--overlap <threshold>] [--report <file>]\n";
}

bool parseGtFormat(const std::string &value, GroundTruthFormat &format)
{
	if (value == "plain_rad")
	{
		format = GroundTruthFormat::PlainRad;
	}
	else if (value == "plain_deg")
	{
		format = GroundTruthFormat::PlainDeg;
	}
	else if (value == "random")
	{
		format = GroundTruthFormat::RandomGt;
	}
	else if (value == "prasad")
	{
		format = GroundTruthFormat::PrasadGt;
	}
	else if (value == "concentric")
	{
		format = GroundTruthFormat::ConcentricGt;
	}
	else
	{
		return false;
	}
	return true;
}

bool parseResultFormat(const std::string &value, ResultFormat &format)
{
	if (value == "aamed_fled")
	{
		format = ResultFormat::AamedFled;
	}
	else if (value == "plain_rad")
	{
		format = ResultFormat::PlainRad;
	}
	else if (value == "plain_deg")
	{
		format = ResultFormat::PlainDeg;
	}
	else
	{
		return false;
	}
	return true;
}

bool parseArgs(int argc, char **argv, EvalOptions &options)
{
	for (int idx = 1; idx < argc; ++idx)
	{
		const std::string arg = argv[idx];
		if (arg == "--dataset-root" && idx + 1 < argc)
		{
			options.datasetRoot = fs::path(argv[++idx]);
		}
		else if (arg == "--imagenames" && idx + 1 < argc)
		{
			options.imageNamesPath = fs::path(argv[++idx]);
		}
		else if (arg == "--gt-dir" && idx + 1 < argc)
		{
			options.gtDir = fs::path(argv[++idx]);
		}
		else if (arg == "--results-dir" && idx + 1 < argc)
		{
			options.resultsDir = fs::path(argv[++idx]);
		}
		else if (arg == "--report" && idx + 1 < argc)
		{
			options.reportPath = fs::path(argv[++idx]);
		}
		else if (arg == "--gt-format" && idx + 1 < argc)
		{
			if (!parseGtFormat(argv[++idx], options.gtFormat))
			{
				std::cerr << "Unknown gt format.\n";
				return false;
			}
		}
		else if (arg == "--result-format" && idx + 1 < argc)
		{
			if (!parseResultFormat(argv[++idx], options.resultFormat))
			{
				std::cerr << "Unknown result format.\n";
				return false;
			}
		}
		else if (arg == "--gt-prefix" && idx + 1 < argc)
		{
			options.gtPrefix = argv[++idx];
		}
		else if (arg == "--gt-suffix" && idx + 1 < argc)
		{
			options.gtSuffix = argv[++idx];
		}
		else if (arg == "--result-suffix" && idx + 1 < argc)
		{
			options.resultSuffix = argv[++idx];
		}
		else if (arg == "--overlap" && idx + 1 < argc)
		{
			options.overlapThreshold = std::stod(argv[++idx]);
		}
		else if (arg == "--help" || arg == "-h")
		{
			printUsage();
			return false;
		}
		else
		{
			std::cerr << "Unknown argument: " << arg << '\n';
			return false;
		}
	}

	if (options.datasetRoot.has_value())
	{
		if (!options.imageNamesPath.has_value())
		{
			options.imageNamesPath = options.datasetRoot.value() / "imagenames.txt";
		}
		if (!options.gtDir.has_value())
		{
			options.gtDir = options.datasetRoot.value() / "gt";
		}
	}

	if (!options.imageNamesPath.has_value() || !options.gtDir.has_value() || !options.resultsDir.has_value())
	{
		printUsage();
		return false;
	}

	return true;
}

std::vector<std::string> readImageNames(const fs::path &path)
{
	std::ifstream in(path);
	if (!in)
	{
		throw std::runtime_error("Failed to open imagenames file: " + path.string());
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

std::vector<double> parseNumbers(const std::string &line)
{
	std::istringstream in(line);
	std::vector<double> values;
	double value = 0.0;
	while (in >> value)
	{
		values.push_back(value);
	}
	return values;
}

EllipseParam convertGroundTruthRecord(const std::vector<double> &raw, GroundTruthFormat format)
{
	EllipseParam ellipse;
	switch (format)
	{
	case GroundTruthFormat::PlainRad:
		ellipse = {raw[0], raw[1], raw[2], raw[3], raw[4], 1.0};
		break;
	case GroundTruthFormat::PlainDeg:
		ellipse = {raw[0], raw[1], raw[2], raw[3], raw[4] / 180.0 * kPi, 1.0};
		break;
	case GroundTruthFormat::RandomGt:
		ellipse = {raw[0] + 1.0, raw[1] + 1.0, raw[2], raw[3], raw[4] / 180.0 * kPi, 1.0};
		break;
	case GroundTruthFormat::PrasadGt:
		ellipse = {raw[0], raw[1], raw[2], raw[3], raw[4], 1.0};
		break;
	case GroundTruthFormat::ConcentricGt:
		ellipse = {raw[1], raw[0], raw[3], raw[2], -raw[4] / 180.0 * kPi, 1.0};
		break;
	}
	return ellipse;
}

std::vector<EllipseParam> readGroundTruthFile(const fs::path &path, GroundTruthFormat format)
{
	std::ifstream in(path);
	if (!in)
	{
		throw std::runtime_error("Failed to open ground-truth file: " + path.string());
	}

	std::string line;
	if (!std::getline(in, line))
	{
		return {};
	}

	const int count = std::stoi(line);
	std::vector<EllipseParam> ellipses;
	ellipses.reserve(std::max(count, 0));
	for (int idx = 0; idx < count && std::getline(in, line); ++idx)
	{
		const auto values = parseNumbers(line);
		if (values.size() < 5)
		{
			continue;
		}
		ellipses.push_back(convertGroundTruthRecord(values, format));
	}
	return ellipses;
}

DetectionFile readDetectionFile(const fs::path &path, ResultFormat format)
{
	DetectionFile detection;
	std::ifstream in(path);
	if (!in)
	{
		return detection;
	}

	std::string line;
	if (!std::getline(in, line))
	{
		return detection;
	}

	if (format == ResultFormat::AamedFled)
	{
		detection.timeMs = std::stod(line);
		while (std::getline(in, line))
		{
			const auto values = parseNumbers(line);
			if (values.size() < 6)
			{
				continue;
			}
			if (static_cast<int>(values[0]) == 2)
			{
				continue;
			}
			detection.ellipses.push_back(
				{values[2] + 1.0, values[1] + 1.0, values[3] / 2.0, values[4] / 2.0, -values[5] / 180.0 * kPi, 1.0});
		}
		return detection;
	}

	const int count = std::stoi(line);
	for (int idx = 0; idx < count && std::getline(in, line); ++idx)
	{
		const auto values = parseNumbers(line);
		if (values.size() < 5)
		{
			continue;
		}
		EllipseParam ellipse = {values[0], values[1], values[2], values[3], values[4], values.size() > 5 ? values[5] : 1.0};
		if (format == ResultFormat::PlainDeg)
		{
			ellipse.thetaRad = ellipse.thetaRad / 180.0 * kPi;
		}
		detection.ellipses.push_back(ellipse);
	}

	if (std::getline(in, line) && !line.empty())
	{
		detection.timeMs = std::stod(line);
	}
	return detection;
}

void ellipseShapeToEquation(const EllipseParam &ellipse, double out[6])
{
	const double xc = ellipse.cx;
	const double yc = ellipse.cy;
	const double a = ellipse.a;
	const double b = ellipse.b;
	const double theta = ellipse.thetaRad;

	const double cosTheta = std::cos(theta);
	const double sinTheta = std::sin(theta);
	const double sin2Theta = std::sin(2 * theta);
	const double cosTheta2 = cosTheta * cosTheta;
	const double sinTheta2 = sinTheta * sinTheta;
	const double aaInv = 1.0 / (a * a);
	const double bbInv = 1.0 / (b * b);

	double params[6];
	params[0] = cosTheta2 * aaInv + sinTheta2 * bbInv;
	params[1] = -0.5 * sin2Theta * (bbInv - aaInv);
	params[2] = cosTheta2 * bbInv + sinTheta2 * aaInv;
	params[3] = (-xc * sinTheta2 + yc * sin2Theta / 2.0) * bbInv - (xc * cosTheta2 + yc * sin2Theta / 2.0) * aaInv;
	params[4] = (-yc * cosTheta2 + xc * sin2Theta / 2.0) * bbInv - (yc * sinTheta2 + xc * sin2Theta / 2.0) * aaInv;
	const double tmp1 = (xc * cosTheta + yc * sinTheta) / a;
	const double tmp2 = (yc * cosTheta - xc * sinTheta) / b;
	params[5] = tmp1 * tmp1 + tmp2 * tmp2 - 1.0;

	const double k = 1.0 / std::sqrt(std::abs(params[0] * params[2] - params[1] * params[1]));
	for (int idx = 0; idx < 6; ++idx)
	{
		out[idx] = params[idx] * k;
	}
}

bool calculateRangeAtY(const double ellipseEquation[6], double y, double *x1, double *x2)
{
	const double A = ellipseEquation[0];
	const double B = ellipseEquation[1];
	const double C = ellipseEquation[2];
	const double D = ellipseEquation[3];
	const double E = ellipseEquation[4];
	const double F = ellipseEquation[5];

	const double delta = std::pow(B * y + D, 2) - A * (C * y * y + 2 * E * y + F);
	if (delta < 0)
	{
		*x1 = 0.0;
		*x2 = -1.0;
		return false;
	}

	*x1 = (-(B * y + D) - std::sqrt(delta)) / A;
	*x2 = (-(B * y + D) + std::sqrt(delta)) / A;
	if (*x2 < *x1)
	{
		std::swap(*x1, *x2);
	}
	return true;
}

double ellipseOverlap(const EllipseParam &lhs, const EllipseParam &rhs)
{
	double lhsEq[6];
	double rhsEq[6];
	ellipseShapeToEquation(lhs, lhsEq);
	ellipseShapeToEquation(rhs, rhsEq);

	const double lhsYMin = lhs.cy - std::max(lhs.a, lhs.b);
	const double lhsYMax = lhs.cy + std::max(lhs.a, lhs.b);
	const double rhsYMin = rhs.cy - std::max(rhs.a, rhs.b);
	const double rhsYMax = rhs.cy + std::max(rhs.a, rhs.b);

	const double yMin = std::floor(std::max(lhsYMin, rhsYMin));
	const double yMax = std::ceil(std::min(lhsYMax, rhsYMax));
	const double searchStep = 0.2;

	if (yMin >= yMax)
	{
		return 0.0;
	}

	double overlapArea = 0.0;
	for (double y = yMin; y <= yMax + 1e-6; y += searchStep)
	{
		double x11 = 0.0, x12 = -1.0, x21 = 0.0, x22 = -1.0;
		const bool lhsValid = calculateRangeAtY(lhsEq, y, &x11, &x12);
		const bool rhsValid = calculateRangeAtY(rhsEq, y, &x21, &x22);
		if (!lhsValid || !rhsValid)
		{
			continue;
		}

		const double xMin = std::max(x11, x21);
		const double xMax = std::min(x12, x22);
		if (xMin < xMax)
		{
			overlapArea += (xMax - xMin);
		}
	}

	const double intersection = overlapArea * searchStep;
	const double unionArea = kPi * lhs.a * lhs.b + kPi * rhs.a * rhs.b - intersection;
	return unionArea <= 0.0 ? 0.0 : intersection / unionArea;
}

std::string formatDouble(double value)
{
	std::ostringstream out;
	out << std::fixed << std::setprecision(6) << value;
	return out.str();
}
}

int main(int argc, char **argv)
{
	EvalOptions options;
	if (!parseArgs(argc, argv, options))
	{
		return 1;
	}

	try
	{
		const std::vector<std::string> imageNames = readImageNames(options.imageNamesPath.value());
		double posAll = 0.0;
		double detAll = 0.0;
		double gtAll = 0.0;
		double timeSum = 0.0;

		for (const auto &imageName : imageNames)
		{
			const fs::path gtPath = options.gtDir.value() / (options.gtPrefix + imageName + options.gtSuffix);
			const fs::path resultPath = options.resultsDir.value() / (imageName + options.resultSuffix);

			const std::vector<EllipseParam> gtEllipses = readGroundTruthFile(gtPath, options.gtFormat);
			const DetectionFile detections = readDetectionFile(resultPath, options.resultFormat);
			timeSum += detections.timeMs;

			std::vector<int> gtMatch(gtEllipses.size(), 0);
			std::vector<int> detMatch(detections.ellipses.size(), 0);
			for (size_t detIdx = 0; detIdx < detections.ellipses.size(); ++detIdx)
			{
				for (size_t gtIdx = 0; gtIdx < gtEllipses.size(); ++gtIdx)
				{
					if (ellipseOverlap(detections.ellipses[detIdx], gtEllipses[gtIdx]) > options.overlapThreshold)
					{
						detMatch[detIdx] += 1;
						gtMatch[gtIdx] += 1;
					}
				}
			}

			const int numLoss = static_cast<int>(std::count(gtMatch.begin(), gtMatch.end(), 0));
			const int numFalse = static_cast<int>(std::count(detMatch.begin(), detMatch.end(), 0));
			const int numTrue = static_cast<int>(std::count_if(gtMatch.begin(), gtMatch.end(), [](int value) { return value > 0; }));
			const int numDetTrue = static_cast<int>(std::count_if(detMatch.begin(), detMatch.end(), [](int value) { return value > 0; }));

			posAll += numTrue;
			detAll += numTrue + numFalse + std::max(numDetTrue - numTrue, 0);
			gtAll += numTrue + numLoss;
		}

		const double precision = detAll > 0.0 ? posAll / detAll : 0.0;
		const double recall = gtAll > 0.0 ? posAll / gtAll : 0.0;
		const double fMeasure = precision + recall > 0.0 ? 2.0 * precision * recall / (precision + recall) : 0.0;
		const double avgTime = imageNames.empty() ? 0.0 : timeSum / static_cast<double>(imageNames.size());

		std::ostringstream report;
		report
			<< "Images: " << imageNames.size() << '\n'
			<< "PositiveMatches: " << posAll << '\n'
			<< "DetectedCount: " << detAll << '\n'
			<< "GroundTruthCount: " << gtAll << '\n'
			<< "Precision: " << formatDouble(precision) << '\n'
			<< "Recall: " << formatDouble(recall) << '\n'
			<< "FMeasure: " << formatDouble(fMeasure) << '\n'
			<< "AverageDetectedTimeMs: " << formatDouble(avgTime) << '\n';

		std::cout << report.str();

		if (options.reportPath.has_value())
		{
			std::ofstream reportOut(options.reportPath.value());
			reportOut << report.str();
		}
	}
	catch (const std::exception &ex)
	{
		std::cerr << "Evaluation failed: " << ex.what() << '\n';
		return 1;
	}

	return 0;
}
