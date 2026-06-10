#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <ostream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

// ==============================
// Unified experiment switches.
// Edit this file to choose:
// 1. run mode
// 2. dataset preset
// 3. enabled optimization methods
// 4. method parameters
// ==============================

#define EXP_RUN_MODE_SINGLE_IMAGE 1
#define EXP_RUN_MODE_DATASET_BATCH 2
#define EXP_RUN_MODE EXP_RUN_MODE_DATASET_BATCH

#define EXP_DATASET_PRESET_PRASAD 1
#define EXP_DATASET_PRESET_CONCENTRIC_SYNTHETIC 2
#define EXP_DATASET_PRESET_CONCURRENT_SYNTHETIC 3
#define EXP_DATASET_PRESET_RANDOM_1 4
#define EXP_DATASET_PRESET_SMARTPHONE_2 5
#define EXP_DATASET_PRESET EXP_DATASET_PRESET_CONCENTRIC_SYNTHETIC

#define EXP_ENABLE_WEIGHTED_ARC 0
#define EXP_ENABLE_MULTI_SCALE_FPN 0
#define EXP_ENABLE_SMALL_ELLIPSE_GUARD 0

namespace experiment
{
namespace fs = std::filesystem;

enum class RunMode
{
	SingleImage,
	DatasetBatch
};

enum class DatasetPreset
{
	Prasad,
	ConcentricSynthetic,
	ConcurrentSynthetic,
	Random1,
	Smartphone2
};

enum class EvalGroundTruthSource
{
	PlainTextWithCount,
	MatlabMatrix
};

enum class EvalEllipseConvention
{
	XYRad,
	XYDeg,
	RowColRad,
	RowColDeg,
	ConcentricDeg,
	SyntheticOccludedDeg
};

enum class EvalMatrixOrientation
{
	Auto,
	RowsAreEllipses,
	ColsAreEllipses
};

enum class EvalResultFormat
{
	AamedFled,
	PlainRad,
	PlainDeg
};

struct DetectorParams
{
	double thetaArc = 3.14159265358979323846 / 3.0;
	double lambdaArc = 3.4;
	double validationThreshold = 0.77;
};

struct WeightedArcConfig
{
	bool enable = EXP_ENABLE_WEIGHTED_ARC != 0;
	float softLinkThreshold = 0.58f;
	int maxSoftNeighborsPerDirection = 8;
	double pairCompatGain = 0.18;
	double crossCompatGain = 0.24;
	double singleCompatGain = 0.22;
	bool selectBestValidatedCandidate = true;
};

struct MultiScaleFpnConfig
{
	bool enable = EXP_ENABLE_MULTI_SCALE_FPN != 0;
	std::vector<int> scales = {2, 3};
	std::vector<double> validationThresholds = {0.77, 0.72};
	double fusionIoU = 0.8;
	int minBranches = 3;
	bool requireCrossScale = true;
	double remapRadiusMax = 20.0;
	bool useCubicInterpolation = true;
};

struct SmallEllipseGuardConfig
{
	bool enable = EXP_ENABLE_SMALL_ELLIPSE_GUARD != 0;
	float minAxisThreshold = 9.0f;
	float missingEdgeCompensation = 0.40f;
	float strongGradientScoreThreshold = 0.72f;
	float minEdgeCoverage = 0.30f;
	float minStrongGradientCoverage = 0.45f;
	bool hardRejectWeakSmallCandidates = true;
};

struct SingleImageConfig
{
	fs::path inputPath;
	fs::path outputDir;
	bool exportDebug = true;
	bool quiet = false;
};

struct DatasetConfig
{
	DatasetPreset preset = DatasetPreset::Prasad;
	fs::path datasetRoot;
	fs::path imagesDir;
	fs::path imageNamesPath;
	fs::path gtDir;
	fs::path resultsDir;
	fs::path evalReportPath;
	bool exportDebug = false;
	bool quiet = true;

	EvalGroundTruthSource groundTruthSource = EvalGroundTruthSource::PlainTextWithCount;
	EvalEllipseConvention groundTruthConvention = EvalEllipseConvention::XYRad;
	EvalMatrixOrientation gtMatrixOrientation = EvalMatrixOrientation::Auto;
	EvalResultFormat resultFormat = EvalResultFormat::AamedFled;
	std::string gtPrefix;
	std::string gtSuffix = ".txt";
	std::string resultSuffix = ".fled.txt";
	double overlapThreshold = 0.8;
};

struct ExperimentConfig
{
	RunMode mode = RunMode::DatasetBatch;
	std::string datasetLabel = "prasad";
	std::string experimentLabel = "baseline";
	DetectorParams detector;
	WeightedArcConfig weightedArc;
	MultiScaleFpnConfig multiScaleFpn;
	SmallEllipseGuardConfig smallEllipseGuard;
	SingleImageConfig single;
	DatasetConfig dataset;
};

inline std::string BuildMethodLabel(
	const WeightedArcConfig &weightedArc,
	const MultiScaleFpnConfig &multiScaleFpn,
	const SmallEllipseGuardConfig &smallEllipseGuard)
{
	std::string label;
	if (weightedArc.enable)
	{
		label = "weighted_arc";
	}
	if (multiScaleFpn.enable)
	{
		if (!label.empty())
		{
			label += "__";
		}
		label += "multi_scale_fpn";
	}
	if (smallEllipseGuard.enable)
	{
		if (!label.empty())
		{
			label += "__";
		}
		label += "small_ellipse_guard";
	}
	return label.empty() ? "baseline" : label;
}

inline std::string DatasetLabel(DatasetPreset preset)
{
	switch (preset)
	{
	case DatasetPreset::Prasad: return "prasad";
	case DatasetPreset::ConcentricSynthetic: return "concentric_synthetic";
	case DatasetPreset::ConcurrentSynthetic: return "concurrent_synthetic";
	case DatasetPreset::Random1: return "random";
	case DatasetPreset::Smartphone2: return "smartphone";
	}
	return "unknown";
}

inline std::string NormalizeToken(std::string value)
{
	std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch)
	{
		return static_cast<char>(std::tolower(ch));
	});
	std::replace(value.begin(), value.end(), '-', '_');
	return value;
}

inline DatasetPreset ParseDatasetPreset(const std::string &value)
{
	const std::string normalized = NormalizeToken(value);
	if (normalized == "prasad") return DatasetPreset::Prasad;
	if (normalized == "random") return DatasetPreset::Random1;
	if (normalized == "concentric" || normalized == "concentric_synthetic") return DatasetPreset::ConcentricSynthetic;
	if (normalized == "concurrent" || normalized == "concurrent_synthetic") return DatasetPreset::ConcurrentSynthetic;
	if (normalized == "smartphone") return DatasetPreset::Smartphone2;
	throw std::runtime_error("Unknown AAMED_DATASET value: " + value);
}

inline void ApplyMethodOverride(const std::string &value, ExperimentConfig &config)
{
	config.weightedArc.enable = false;
	config.multiScaleFpn.enable = false;
	config.smallEllipseGuard.enable = false;

	std::istringstream in(value);
	std::string token;
	while (std::getline(in, token, ','))
	{
		token = NormalizeToken(token);
		token.erase(std::remove_if(token.begin(), token.end(), [](unsigned char ch) { return std::isspace(ch) != 0; }), token.end());
		if (token.empty() || token == "baseline")
		{
			continue;
		}
		if (token == "weighted_arc")
		{
			config.weightedArc.enable = true;
		}
		else if (token == "multi_scale_fpn")
		{
			config.multiScaleFpn.enable = true;
		}
		else if (token == "small_ellipse_guard")
		{
			config.smallEllipseGuard.enable = true;
		}
		else
		{
			throw std::runtime_error("Unknown AAMED_METHODS value: " + token);
		}
	}
}

inline DatasetConfig BuildDatasetConfig(
	const fs::path &projectRoot,
	DatasetPreset preset,
	const std::string &datasetLabel,
	const std::string &experimentLabel)
{
	DatasetConfig config;
	config.preset = preset;
	config.resultsDir = projectRoot / "output" / datasetLabel / experimentLabel;
	config.evalReportPath = config.resultsDir / "eval_report.txt";

	switch (preset)
	{
	case DatasetPreset::Prasad:
		config.datasetRoot = projectRoot / "dataset" / "Prasad Images - Dataset Prasad";
		config.imagesDir = config.datasetRoot / "images";
		config.imageNamesPath = config.datasetRoot / "imagenames.txt";
		config.gtDir = config.datasetRoot / "gt";
		config.groundTruthSource = EvalGroundTruthSource::PlainTextWithCount;
		config.groundTruthConvention = EvalEllipseConvention::XYRad;
		config.gtMatrixOrientation = EvalMatrixOrientation::Auto;
		config.resultFormat = EvalResultFormat::AamedFled;
		config.gtPrefix = "gt_";
		config.gtSuffix = ".txt";
		config.resultSuffix = ".fled.txt";
		config.overlapThreshold = 0.8;
		break;
	case DatasetPreset::ConcentricSynthetic:
		config.datasetRoot = projectRoot / "dataset" / "Concentric Ellipses - Dataset Synthetic";
		config.imagesDir = config.datasetRoot / "images";
		config.imageNamesPath = config.datasetRoot / "imagenames.txt";
		config.gtDir = config.datasetRoot / "gt";
		config.groundTruthSource = EvalGroundTruthSource::PlainTextWithCount;
		config.groundTruthConvention = EvalEllipseConvention::ConcentricDeg;
		config.gtMatrixOrientation = EvalMatrixOrientation::Auto;
		config.resultFormat = EvalResultFormat::AamedFled;
		config.gtPrefix.clear();
		config.gtSuffix = ".txt";
		config.resultSuffix = ".fled.txt";
		config.overlapThreshold = 0.95;
		break;
	case DatasetPreset::ConcurrentSynthetic:
		config.datasetRoot = projectRoot / "dataset" / "Concurrent Ellipses - Dataset Synthetic";
		config.imagesDir = config.datasetRoot / "images";
		config.imageNamesPath = config.datasetRoot / "imagenames.txt";
		config.gtDir = config.datasetRoot / "gt";
		config.groundTruthSource = EvalGroundTruthSource::PlainTextWithCount;
		config.groundTruthConvention = EvalEllipseConvention::ConcentricDeg;
		config.gtMatrixOrientation = EvalMatrixOrientation::Auto;
		config.resultFormat = EvalResultFormat::AamedFled;
		config.gtPrefix.clear();
		config.gtSuffix = ".txt";
		config.resultSuffix = ".fled.txt";
		config.overlapThreshold = 0.95;
		break;
	case DatasetPreset::Random1:
		config.datasetRoot = projectRoot / "dataset" / "Random Images - Dataset #1";
		config.imagesDir = config.datasetRoot / "images";
		config.imageNamesPath = config.datasetRoot / "imagenames.txt";
		config.gtDir = config.datasetRoot / "gt";
		config.groundTruthSource = EvalGroundTruthSource::PlainTextWithCount;
		config.groundTruthConvention = EvalEllipseConvention::XYDeg;
		config.gtMatrixOrientation = EvalMatrixOrientation::Auto;
		config.resultFormat = EvalResultFormat::AamedFled;
		config.gtPrefix = "gt_";
		config.gtSuffix = ".txt";
		config.resultSuffix = ".fled.txt";
		config.overlapThreshold = 0.8;
		break;
	case DatasetPreset::Smartphone2:
		config.datasetRoot = projectRoot / "dataset" / "Smartphone Images - Dataset #2";
		config.imagesDir = config.datasetRoot / "images";
		config.imageNamesPath = config.datasetRoot / "imagenames.txt";
		config.gtDir = config.datasetRoot / "gt";
		config.groundTruthSource = EvalGroundTruthSource::PlainTextWithCount;
		config.groundTruthConvention = EvalEllipseConvention::XYDeg;
		config.gtMatrixOrientation = EvalMatrixOrientation::Auto;
		config.resultFormat = EvalResultFormat::AamedFled;
		config.gtPrefix = "gt_";
		config.gtSuffix = ".txt";
		config.resultSuffix = ".fled.txt";
		config.overlapThreshold = 0.8;
		break;
	}

	return config;
}

inline ExperimentConfig BuildExperimentConfig()
{
	const fs::path projectRoot(AAMED_OPENCV_PROJECT_ROOT);

	ExperimentConfig config;
#if EXP_RUN_MODE == EXP_RUN_MODE_SINGLE_IMAGE
	config.mode = RunMode::SingleImage;
#else
	config.mode = RunMode::DatasetBatch;
#endif

#if EXP_DATASET_PRESET == EXP_DATASET_PRESET_CONCENTRIC_SYNTHETIC
	DatasetPreset datasetPreset = DatasetPreset::ConcentricSynthetic;
#elif EXP_DATASET_PRESET == EXP_DATASET_PRESET_CONCURRENT_SYNTHETIC
	DatasetPreset datasetPreset = DatasetPreset::ConcurrentSynthetic;
#elif EXP_DATASET_PRESET == EXP_DATASET_PRESET_RANDOM_1
	DatasetPreset datasetPreset = DatasetPreset::Random1;
#elif EXP_DATASET_PRESET == EXP_DATASET_PRESET_SMARTPHONE_2
	DatasetPreset datasetPreset = DatasetPreset::Smartphone2;
#else
	DatasetPreset datasetPreset = DatasetPreset::Prasad;
#endif

	if (const char *datasetOverride = std::getenv("AAMED_DATASET"))
	{
		datasetPreset = ParseDatasetPreset(datasetOverride);
	}
	if (const char *methodOverride = std::getenv("AAMED_METHODS"))
	{
		ApplyMethodOverride(methodOverride, config);
	}

	config.datasetLabel = DatasetLabel(datasetPreset);
	config.experimentLabel = BuildMethodLabel(
		config.weightedArc,
		config.multiScaleFpn,
		config.smallEllipseGuard);
	config.dataset = BuildDatasetConfig(projectRoot, datasetPreset, config.datasetLabel, config.experimentLabel);
	config.single.inputPath = projectRoot / "demo" / "033_0053.jpg";
	config.single.outputDir = projectRoot / "output" / "single" / config.datasetLabel / config.experimentLabel;

	return config;
}

inline bool ShouldPrintConfig()
{
	const char *value = std::getenv("AAMED_PRINT_CONFIG");
	return value != nullptr && std::string(value) == "1";
}

inline void PrintExperimentConfig(const ExperimentConfig &config, std::ostream &out)
{
	out << "Dataset: " << config.datasetLabel << '\n';
	out << "Experiment: " << config.experimentLabel << '\n';
	out << "WeightedArc: " << (config.weightedArc.enable ? 1 : 0) << '\n';
	out << "MultiScaleFpn: " << (config.multiScaleFpn.enable ? 1 : 0) << '\n';
	out << "SmallEllipseGuard: " << (config.smallEllipseGuard.enable ? 1 : 0) << '\n';
	out << "DatasetRoot: " << config.dataset.datasetRoot.string() << '\n';
	out << "ResultsDir: " << config.dataset.resultsDir.string() << '\n';
	out << "GtPrefix: " << config.dataset.gtPrefix << '\n';
	out << "OverlapThreshold: " << config.dataset.overlapThreshold << '\n';
	out << "GroundTruthConvention: " << static_cast<int>(config.dataset.groundTruthConvention) << '\n';
}
}
