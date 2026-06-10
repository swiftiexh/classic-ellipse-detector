#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../src/ExperimentConfig.h"

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
	double score = 1.0;
};

struct DetectionFile
{
	double timeMs = 0.0;
	std::vector<EllipseParam> ellipses;
};

enum class GroundTruthSource
{
	PlainTextWithCount,
	MatlabMatrix
};

enum class EllipseConvention
{
	XYRad,
	XYDeg,
	RowColRad,
	RowColDeg,
	ConcentricDeg,
	SyntheticOccludedDeg
};

enum class MatrixOrientation
{
	Auto,
	RowsAreEllipses,
	ColsAreEllipses
};

enum class ResultFormat
{
	AamedFled,
	PlainRad,
	PlainDeg
};

struct EvalConfig
{
	fs::path datasetRoot = fs::path(AAMED_OPENCV_PROJECT_ROOT) / "dataset";
	fs::path imagesDir = datasetRoot / "images";
	fs::path imageNamesPath = datasetRoot / "imagenames.txt";
	fs::path gtDir = datasetRoot / "gt";
	fs::path resultsDir = fs::path(AAMED_OPENCV_PROJECT_ROOT) / "output" / "dataset";
	fs::path reportPath = resultsDir / "prasad_eval_report.txt";

	GroundTruthSource groundTruthSource = GroundTruthSource::PlainTextWithCount;
	EllipseConvention groundTruthConvention = EllipseConvention::XYRad;
	MatrixOrientation gtMatrixOrientation = MatrixOrientation::Auto;
	ResultFormat resultFormat = ResultFormat::AamedFled;

	std::string gtPrefix = "gt_";
	std::string gtSuffix = ".txt";
	std::string resultSuffix = ".fled.txt";
	double overlapThreshold = 0.8;
};

EvalConfig buildConfig()
{
	const experiment::ExperimentConfig shared = experiment::BuildExperimentConfig();

	auto mapGroundTruthSource = [](experiment::EvalGroundTruthSource source)
	{
		switch (source)
		{
		case experiment::EvalGroundTruthSource::PlainTextWithCount:
			return GroundTruthSource::PlainTextWithCount;
		case experiment::EvalGroundTruthSource::MatlabMatrix:
			return GroundTruthSource::MatlabMatrix;
		}
		return GroundTruthSource::PlainTextWithCount;
	};

	auto mapConvention = [](experiment::EvalEllipseConvention convention)
	{
		switch (convention)
		{
		case experiment::EvalEllipseConvention::XYRad:
			return EllipseConvention::XYRad;
		case experiment::EvalEllipseConvention::XYDeg:
			return EllipseConvention::XYDeg;
		case experiment::EvalEllipseConvention::RowColRad:
			return EllipseConvention::RowColRad;
		case experiment::EvalEllipseConvention::RowColDeg:
			return EllipseConvention::RowColDeg;
		case experiment::EvalEllipseConvention::ConcentricDeg:
			return EllipseConvention::ConcentricDeg;
		case experiment::EvalEllipseConvention::SyntheticOccludedDeg:
			return EllipseConvention::SyntheticOccludedDeg;
		}
		return EllipseConvention::XYRad;
	};

	auto mapOrientation = [](experiment::EvalMatrixOrientation orientation)
	{
		switch (orientation)
		{
		case experiment::EvalMatrixOrientation::Auto:
			return MatrixOrientation::Auto;
		case experiment::EvalMatrixOrientation::RowsAreEllipses:
			return MatrixOrientation::RowsAreEllipses;
		case experiment::EvalMatrixOrientation::ColsAreEllipses:
			return MatrixOrientation::ColsAreEllipses;
		}
		return MatrixOrientation::Auto;
	};

	auto mapResultFormat = [](experiment::EvalResultFormat format)
	{
		switch (format)
		{
		case experiment::EvalResultFormat::AamedFled:
			return ResultFormat::AamedFled;
		case experiment::EvalResultFormat::PlainRad:
			return ResultFormat::PlainRad;
		case experiment::EvalResultFormat::PlainDeg:
			return ResultFormat::PlainDeg;
		}
		return ResultFormat::AamedFled;
	};

	EvalConfig config;
	config.datasetRoot = shared.dataset.datasetRoot;
	config.imagesDir = shared.dataset.imagesDir;
	config.imageNamesPath = shared.dataset.imageNamesPath;
	config.gtDir = shared.dataset.gtDir;
	config.resultsDir = shared.dataset.resultsDir;
	config.reportPath = shared.dataset.evalReportPath;
	config.groundTruthSource = mapGroundTruthSource(shared.dataset.groundTruthSource);
	config.groundTruthConvention = mapConvention(shared.dataset.groundTruthConvention);
	config.gtMatrixOrientation = mapOrientation(shared.dataset.gtMatrixOrientation);
	config.resultFormat = mapResultFormat(shared.dataset.resultFormat);
	config.gtPrefix = shared.dataset.gtPrefix;
	config.gtSuffix = shared.dataset.gtSuffix;
	config.resultSuffix = shared.dataset.resultSuffix;
	config.overlapThreshold = shared.dataset.overlapThreshold;
	return config;
}

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

fs::path resolveGroundTruthPath(const EvalConfig &config, const fs::path &resolvedImagePath, const std::string &imageName)
{
	const std::array<std::string, 2> bases = {
		resolvedImagePath.filename().string(),
		resolvedImagePath.stem().string()
	};

	for (const std::string &base : bases)
	{
		const fs::path candidate = config.gtDir / (config.gtPrefix + base + config.gtSuffix);
		if (fs::exists(candidate))
		{
			return candidate;
		}
	}

	return config.gtDir / (config.gtPrefix + imageName + config.gtSuffix);
}

fs::path resolveResultPath(const EvalConfig &config, const fs::path &resolvedImagePath, const std::string &imageName)
{
	const fs::path byResolvedName = config.resultsDir / (resolvedImagePath.filename().string() + config.resultSuffix);
	if (fs::exists(byResolvedName))
	{
		return byResolvedName;
	}

	return config.resultsDir / (imageName + config.resultSuffix);
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

EllipseParam convertRawEllipseRecord(const std::vector<double> &raw, EllipseConvention convention)
{
	EllipseParam ellipse;
	switch (convention)
	{
	case EllipseConvention::XYRad:
		ellipse = {raw[0], raw[1], raw[2], raw[3], raw[4], 1.0};
		break;
	case EllipseConvention::XYDeg:
		ellipse = {raw[0], raw[1], raw[2], raw[3], raw[4] / 180.0 * kPi, 1.0};
		break;
	case EllipseConvention::RowColRad:
		ellipse = {raw[1], raw[0], raw[2], raw[3], raw[4], 1.0};
		break;
	case EllipseConvention::RowColDeg:
		ellipse = {raw[1], raw[0], raw[2], raw[3], raw[4] / 180.0 * kPi, 1.0};
		break;
	case EllipseConvention::ConcentricDeg:
		ellipse = {raw[1], raw[0], raw[3], raw[2], -raw[4] / 180.0 * kPi, 1.0};
		break;
	case EllipseConvention::SyntheticOccludedDeg:
		ellipse = {raw[1], raw[0], raw[3], raw[2], kPi / 2.0 - raw[4] / 180.0 * kPi, 1.0};
		break;
	}
	return ellipse;
}

std::vector<EllipseParam> readPlainTextGroundTruthFile(const fs::path &path, EllipseConvention convention)
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
		if (values.size() >= 5)
		{
			ellipses.push_back(convertRawEllipseRecord(values, convention));
		}
	}
	return ellipses;
}

std::vector<std::uint8_t> readBinaryFile(const fs::path &path)
{
	std::ifstream in(path, std::ios::binary);
	if (!in)
	{
		throw std::runtime_error("Failed to open binary file: " + path.string());
	}

	in.seekg(0, std::ios::end);
	const std::streamsize size = in.tellg();
	in.seekg(0, std::ios::beg);

	if (size <= 0)
	{
		return {};
	}

	std::vector<std::uint8_t> bytes(static_cast<size_t>(size));
	in.read(reinterpret_cast<char *>(bytes.data()), size);
	return bytes;
}

struct DataElement
{
	std::uint32_t type = 0;
	std::uint32_t bytes = 0;
	size_t payloadOffset = 0;
	size_t nextOffset = 0;
};

size_t align8(size_t value)
{
	return (value + 7u) & ~size_t(7u);
}

std::uint16_t readU16LE(const std::vector<std::uint8_t> &bytes, size_t offset)
{
	return static_cast<std::uint16_t>(bytes[offset])
		| (static_cast<std::uint16_t>(bytes[offset + 1]) << 8);
}

std::uint32_t readU32LE(const std::vector<std::uint8_t> &bytes, size_t offset)
{
	return static_cast<std::uint32_t>(bytes[offset])
		| (static_cast<std::uint32_t>(bytes[offset + 1]) << 8)
		| (static_cast<std::uint32_t>(bytes[offset + 2]) << 16)
		| (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
}

std::int32_t readI32LE(const std::vector<std::uint8_t> &bytes, size_t offset)
{
	return static_cast<std::int32_t>(readU32LE(bytes, offset));
}

std::uint64_t readU64LE(const std::vector<std::uint8_t> &bytes, size_t offset)
{
	return static_cast<std::uint64_t>(readU32LE(bytes, offset))
		| (static_cast<std::uint64_t>(readU32LE(bytes, offset + 4)) << 32);
}

double readF64LE(const std::vector<std::uint8_t> &bytes, size_t offset)
{
	double value = 0.0;
	std::uint64_t raw = readU64LE(bytes, offset);
	std::memcpy(&value, &raw, sizeof(double));
	return value;
}

float readF32LE(const std::vector<std::uint8_t> &bytes, size_t offset)
{
	float value = 0.0f;
	std::uint32_t raw = readU32LE(bytes, offset);
	std::memcpy(&value, &raw, sizeof(float));
	return value;
}

bool readDataElement(const std::vector<std::uint8_t> &bytes, size_t offset, DataElement &element)
{
	if (offset + 8 > bytes.size())
	{
		return false;
	}

	const std::uint16_t packedBytes = readU16LE(bytes, offset + 2);
	if (packedBytes != 0)
	{
		element.type = readU16LE(bytes, offset);
		element.bytes = packedBytes;
		element.payloadOffset = offset + 4;
		element.nextOffset = offset + 8;
		return element.payloadOffset + element.bytes <= bytes.size();
	}

	element.type = readU32LE(bytes, offset);
	element.bytes = readU32LE(bytes, offset + 4);
	element.payloadOffset = offset + 8;
	element.nextOffset = align8(element.payloadOffset + element.bytes);
	return element.payloadOffset + element.bytes <= bytes.size() && element.nextOffset <= bytes.size();
}

struct NumericMatrix
{
	size_t rows = 0;
	size_t cols = 0;
	std::vector<double> values;

	double at(size_t row, size_t col) const
	{
		return values[col * rows + row];
	}
};

double readNumericValue(const std::vector<std::uint8_t> &bytes, std::uint32_t type, size_t offset)
{
	switch (type)
	{
	case 1:
		return static_cast<double>(static_cast<std::int8_t>(bytes[offset]));
	case 2:
		return static_cast<double>(bytes[offset]);
	case 3:
		return static_cast<double>(static_cast<std::int16_t>(readU16LE(bytes, offset)));
	case 4:
		return static_cast<double>(readU16LE(bytes, offset));
	case 5:
		return static_cast<double>(readI32LE(bytes, offset));
	case 6:
		return static_cast<double>(readU32LE(bytes, offset));
	case 7:
		return static_cast<double>(readF32LE(bytes, offset));
	case 9:
		return readF64LE(bytes, offset);
	case 12:
		return static_cast<double>(static_cast<std::int64_t>(readU64LE(bytes, offset)));
	case 13:
		return static_cast<double>(readU64LE(bytes, offset));
	default:
		throw std::runtime_error("Unsupported MATLAB numeric type: " + std::to_string(type));
	}
}

size_t matlabTypeSize(std::uint32_t type)
{
	switch (type)
	{
	case 1:
	case 2:
		return 1;
	case 3:
	case 4:
		return 2;
	case 5:
	case 6:
	case 7:
		return 4;
	case 9:
	case 12:
	case 13:
		return 8;
	default:
		throw std::runtime_error("Unsupported MATLAB numeric type size: " + std::to_string(type));
	}
}

std::optional<NumericMatrix> extractFirstMatlabMatrix(const std::vector<std::uint8_t> &bytes)
{
	if (bytes.size() < 128)
	{
		return std::nullopt;
	}

	const std::string header(reinterpret_cast<const char *>(bytes.data()),
		reinterpret_cast<const char *>(bytes.data()) + std::min<size_t>(116, bytes.size()));
	if (header.rfind("MATLAB 5.0 MAT-file", 0) != 0)
	{
		throw std::runtime_error("Unsupported MAT header. Only MATLAB 5 MAT files are supported.");
	}

	if (bytes[126] != 'I' || bytes[127] != 'M')
	{
		throw std::runtime_error("Unsupported MAT endianness. Only little-endian files are supported.");
	}

	size_t offset = 128;
	while (offset + 8 <= bytes.size())
	{
		DataElement element;
		if (!readDataElement(bytes, offset, element))
		{
			break;
		}

		if (element.type == 15)
		{
			throw std::runtime_error("Compressed MAT files are not supported by this evaluator.");
		}

		if (element.type == 14)
		{
			size_t innerOffset = element.payloadOffset;
			size_t rows = 0;
			size_t cols = 0;
			bool isComplex = false;
			std::uint32_t realType = 0;
			size_t realOffset = 0;
			size_t realBytes = 0;

			while (innerOffset + 8 <= element.payloadOffset + element.bytes)
			{
				DataElement subElement;
				if (!readDataElement(bytes, innerOffset, subElement))
				{
					break;
				}

				if (subElement.type == 6 && subElement.bytes >= 8)
				{
					const std::uint32_t flags = readU32LE(bytes, subElement.payloadOffset);
					isComplex = (flags & 0x0800u) != 0;
				}
				else if (subElement.type == 5 && subElement.bytes >= 8)
				{
					rows = static_cast<size_t>(readI32LE(bytes, subElement.payloadOffset));
					cols = static_cast<size_t>(readI32LE(bytes, subElement.payloadOffset + 4));
				}
				else if (subElement.type != 1 && subElement.type != 2)
				{
					realType = subElement.type;
					realOffset = subElement.payloadOffset;
					realBytes = subElement.bytes;
					break;
				}

				innerOffset = subElement.nextOffset;
			}

			if (!isComplex && rows > 0 && cols > 0 && realType != 0)
			{
				const size_t count = rows * cols;
				const size_t valueSize = matlabTypeSize(realType);
				if (count * valueSize > realBytes)
				{
					throw std::runtime_error("MAT matrix payload is smaller than expected.");
				}

				NumericMatrix matrix;
				matrix.rows = rows;
				matrix.cols = cols;
				matrix.values.resize(count);
				for (size_t idx = 0; idx < count; ++idx)
				{
					matrix.values[idx] = readNumericValue(bytes, realType, realOffset + idx * valueSize);
				}
				return matrix;
			}
		}

		offset = element.nextOffset;
	}

	return std::nullopt;
}

std::vector<EllipseParam> convertMatrixToEllipses(
	const NumericMatrix &matrix,
	EllipseConvention convention,
	MatrixOrientation orientation)
{
	std::vector<EllipseParam> ellipses;
	bool rowsAreEllipses = true;

	if (orientation == MatrixOrientation::RowsAreEllipses)
	{
		rowsAreEllipses = true;
	}
	else if (orientation == MatrixOrientation::ColsAreEllipses)
	{
		rowsAreEllipses = false;
	}
	else if (matrix.cols == 5)
	{
		rowsAreEllipses = true;
	}
	else if (matrix.rows == 5)
	{
		rowsAreEllipses = false;
	}
	else if (matrix.cols >= 5)
	{
		rowsAreEllipses = true;
	}
	else if (matrix.rows >= 5)
	{
		rowsAreEllipses = false;
	}
	else
	{
		throw std::runtime_error("MAT ground-truth matrix does not contain enough columns for ellipse parameters.");
	}

	if (rowsAreEllipses)
	{
		for (size_t row = 0; row < matrix.rows; ++row)
		{
			if (matrix.cols < 5)
			{
				break;
			}
			const std::vector<double> raw = {
				matrix.at(row, 0),
				matrix.at(row, 1),
				matrix.at(row, 2),
				matrix.at(row, 3),
				matrix.at(row, 4)
			};
			ellipses.push_back(convertRawEllipseRecord(raw, convention));
		}
	}
	else
	{
		for (size_t col = 0; col < matrix.cols; ++col)
		{
			if (matrix.rows < 5)
			{
				break;
			}
			const std::vector<double> raw = {
				matrix.at(0, col),
				matrix.at(1, col),
				matrix.at(2, col),
				matrix.at(3, col),
				matrix.at(4, col)
			};
			ellipses.push_back(convertRawEllipseRecord(raw, convention));
		}
	}

	return ellipses;
}

std::vector<EllipseParam> readMatlabGroundTruthFile(
	const fs::path &path,
	EllipseConvention convention,
	MatrixOrientation orientation)
{
	const std::vector<std::uint8_t> bytes = readBinaryFile(path);
	const std::optional<NumericMatrix> matrix = extractFirstMatlabMatrix(bytes);
	if (!matrix.has_value())
	{
		throw std::runtime_error("No numeric matrix was found in ground-truth MAT file: " + path.string());
	}
	return convertMatrixToEllipses(matrix.value(), convention, orientation);
}

std::vector<EllipseParam> readGroundTruthFile(const fs::path &path, const EvalConfig &config)
{
	if (config.groundTruthSource == GroundTruthSource::PlainTextWithCount)
	{
		return readPlainTextGroundTruthFile(path, config.groundTruthConvention);
	}
	return readMatlabGroundTruthFile(path, config.groundTruthConvention, config.gtMatrixOrientation);
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
		double x11 = 0.0;
		double x12 = -1.0;
		double x21 = 0.0;
		double x22 = -1.0;
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

int main()
{
	try
	{
		const experiment::ExperimentConfig sharedConfig = experiment::BuildExperimentConfig();
		if (experiment::ShouldPrintConfig())
		{
			experiment::PrintExperimentConfig(sharedConfig, std::cout);
			return 0;
		}
		const EvalConfig config = buildConfig();
		std::vector<std::string> imageNames = readImageNames(config.imageNamesPath);
		if (imageNames.empty())
		{
			imageNames = enumerateImageNames(config.imagesDir);
		}

		double posAll = 0.0;
		double detAll = 0.0;
		double gtAll = 0.0;
		double timeSum = 0.0;

		for (const std::string &imageName : imageNames)
		{
			const fs::path resolvedImagePath = resolveImagePath(config.imagesDir, imageName);
			const fs::path gtPath = resolveGroundTruthPath(config, resolvedImagePath, imageName);
			const fs::path resultPath = resolveResultPath(config, resolvedImagePath, imageName);

			const std::vector<EllipseParam> gtEllipses = readGroundTruthFile(gtPath, config);
			const DetectionFile detections = readDetectionFile(resultPath, config.resultFormat);
			timeSum += detections.timeMs;

			std::vector<int> gtMatch(gtEllipses.size(), 0);
			std::vector<int> detMatch(detections.ellipses.size(), 0);
			for (size_t detIdx = 0; detIdx < detections.ellipses.size(); ++detIdx)
			{
				for (size_t gtIdx = 0; gtIdx < gtEllipses.size(); ++gtIdx)
				{
					if (ellipseOverlap(detections.ellipses[detIdx], gtEllipses[gtIdx]) > config.overlapThreshold)
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

		fs::create_directories(config.reportPath.parent_path());
		std::ofstream reportOut(config.reportPath);
		reportOut << report.str();
	}
	catch (const std::exception &ex)
	{
		std::cerr << "Evaluation failed: " << ex.what() << '\n';
		return 1;
	}

	return 0;
}
