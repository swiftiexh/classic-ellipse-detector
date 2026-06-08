#include "FLED.h"
#include "adaptApproximateContours.h"
#include "EllipseNonMaximumSuppression.h"

#include <algorithm>
#include <cmath>

FLED::FLED(int drows, int dcols)
{
	
	_T_dp = sqrt(2.0);
	_T_gradnum = 5;


	vld_use_time = 0;

	dROWS = drows; dCOLS = dcols;
	data = new Node_FC[dROWS*dCOLS];
	for (int i = 0; i < dROWS; i++)
		for (int j = 0; j < dCOLS; j++)
			data[dIDX(i, j)] = Node_FC(i, j, PIXEL_SCALE);

	dls_C = Mat::zeros(6, 6, CV_64FC1);
	//dls_C.at<double>(0, 2) = 0.5;
	//dls_C.at<double>(1, 1) = -1;
	//dls_C.at<double>(2, 0) = 0.5;
	dls_C.at<double>(2, 2) = 1;


	double h = CV_PI * 2 / VALIDATION_NUMBER;
	for (int i = 0; i < VALIDATION_NUMBER; i++)
	{
		vldBaseData[i][0] = cos(i*h);
		vldBaseData[i][1] = sin(i*h);
		vldBaseDataX[i] = cos(i*h);
		vldBaseDataY[i] = sin(i*h);
	}

	LinkMatrix.Update(800 * 800);
	LinkWeight.Update(800 * 800);
	lA.resize(800);

	visited.Update(800);
	fitComb_LR.Update(512);
	asrs.reserve(800);
	ArcFitMat.reserve(512);
	///////Pre-Reserver////////////////////////
	query_arcs.resize(2);
	//oneContour.reserve(1000);
	//oneContourOpp.reserve(1000);
	//candidateLines.reserve(1000);
	//detEllipse.reserve(300);
#ifdef DETAIL_BREAKDOWN
	fitting_time = valitation_time = 0;
	t_allfitting = t_allvalidation = 0;


	dt_time = ds_t_preprocess = ds_t_arcsegment = ds_t_cluster = ds_t_allgrouping = ds_t_allfitting = ds_t_allvalidation = 0;
	ds_fitting_time = ds_valitation_time = ds_num = 0;


#endif

}

void FLED::initFrame()
{
	edgeContours.clear();
	dpContours.clear();
	FSA_ArcContours.clear();

	detEllipses.clear();
	detEllipseScore.clear();


	StrongArcs.clear();
	WeakArcs.clear();
}
void FLED::release()
{
	if (data != NULL)
		delete[] data;

	LinkMatrix.release();
	LinkWeight.release();
	visited.release();
//	lA.release();

}
bool FLED::checkInputImage(int rows, int cols)
{
    if (rows > dROWS || cols > dCOLS)
        return false;
    else
        return true;
}
void FLED::run_AAMED_WithoutCanny(Mat Img_G)
{

	// Pre-Processing
#ifdef DETAIL_BREAKDOWN
	double tmp_st;
	tmp_st = cv::getTickCount();
#endif

	double t1, t2;

	sum_time = 0; fitnum = 0;
	case_stat[0] = case_stat[1] = case_stat[2] = case_stat[3] = 0;
	//isBIDIR = 0;

	initFrame();

	//t1 = cv::getTickCount();
	iROWS = Img_G.rows; iCOLS = Img_G.cols;

	imgCanny = Img_G.clone();

	//t1 = cv::getTickCount();
	uchar *_imgCanny = (uchar*)imgCanny.data;
	//边缘轮廓查找，个数小于T_edge_num(参数自适应)将会被扔掉
	findContours(_imgCanny);


	BoldEdge(_imgCanny, edgeContours); //将图像进行加粗，方便查找边缘点
#ifdef DETAIL_BREAKDOWN
	t_preprocess = (cv::getTickCount() - tmp_st) * 1000 / cv::getTickFrequency();

	tmp_st = cv::getTickCount();
#endif

	// Arc Segmentation
	const int edgeContoursNum = int(edgeContours.size());
	for (int i = 0; i < edgeContoursNum; i++)
	{
#if ADAPT_APPROX_CONTOURS
		//adaptApproximateContours(edgeContours[i], oneContour);
		adaptApproxPolyDP(edgeContours[i], oneContour);
#else
		approxPolyDP(edgeContours[i], oneContour, _T_dp, false);
#endif
		if (oneContour.size() < MIN_DP_CONTOUR_NUM)
			continue;
		dpContours.push_back(oneContour);
	}
	//t2 = cv::getTickCount();
	//cout << "RDP using time:" << (t2 - t1) * 1000 / cv::getTickFrequency() << endl;

	//drawDPContours();


	const int dpContoursNum = int(dpContours.size());
	for (int i = 0; i < dpContoursNum; i++)
	{
		FSA_Segment(dpContours[i]);
	}

#ifdef DETAIL_BREAKDOWN
	t_arcsegment = (cv::getTickCount() - tmp_st) * 1000 / cv::getTickFrequency();
	tmp_st = cv::getTickCount();
#endif


	SortArcs(FSA_ArcContours); // 针对弧段个数进行排序

	CreateArcSearchRegion(FSA_ArcContours); //计算所有弧段的搜索区间

	int fsa_Arcs = int(FSA_ArcContours.size());
	if (fsa_Arcs == 0)
		return;
	getArcs_KDTrees(FSA_ArcContours);


	Arcs_Grouping(FSA_ArcContours);

	//writeLinkMatrix();
	//t2 = cv::getTickCount();
	//sum_time += (t2 - t1) * 1000 / cv::getTickFrequency();

	const char *_linkMatrix = LinkMatrix.GetDataPoint();
	getlinkArcs(_linkMatrix, fsa_Arcs);

	//drawFSA_ArcContours();
	//cv::waitKey(1);


	visited.Update(fsa_Arcs);
	searched.Update(fsa_Arcs);



	GetArcMat(ArcFitMat, FSA_ArcContours); //计算每个弧段的拟合矩阵


	_arc_grouped_label.clear();
	_arc_grouped_label.resize(fsa_Arcs, 0);


	for (int root_idx = 0; root_idx < fsa_Arcs; root_idx++)
	{
		if (FSA_ArcContours[root_idx].size() <= 3)
			continue;

		if (_arc_grouped_label[root_idx] != 0)
			continue;
		//t1 = cv::getTickCount();
		search_group[0].clear(), search_group[1].clear();//清空上一租数据点
		search_arcMats[0].clear(), search_arcMats[1].clear(); //清空上一组矩阵拟合数据点
		searched.clean();

		temp.push_back(root_idx);//加入根节点弧段
		*visited[root_idx] = 1; //当前节点访问过
		memcpy(fitArcTemp.val, ArcFitMat[root_idx].val, sizeof(double)*MAT_NUMBER);

		//t1 = cv::getTickCount();
		PosteriorArcsSearch2(root_idx);
		//t2 = (double)cv::getTickCount();



		temp.push_back(root_idx);
		*visited[root_idx] = 1;
		AnteriorArcsSearch(root_idx);

		//t1 = cv::getTickCount();
		sortCombine(fitComb_LR, search_arcMats, search_group);
		//t2 = (double)cv::getTickCount();
		//printf("The %dth root arc: the search_group size is %d, %d\n", root_idx, search_group[0].size(), search_group[1].size());
		//t1 = cv::getTickCount();
		BiDirectionVerification(fitComb_LR, search_arcMats, search_group, _arc_grouped_label);
		//t2 = (double)cv::getTickCount();
		//sum_time += (t2 - t1) * 1000 / cv::getTickFrequency();
	}

#ifdef DETAIL_BREAKDOWN
	t_allgrouping = (cv::getTickCount() - tmp_st) * 1000 / cv::getTickFrequency();
	tmp_st = cv::getTickCount();
#endif


#if SELECT_CLUSTER_METHOD == OUR_CLUSTER_METHOD
	EllipseNonMaximumSuppression(detEllipses, detEllipseScore, 0.7);
#elif SELECT_CLUSTER_METHOD == PRASAD_CLUSTER_METHOD
	ClusterEllipses(detEllipses, detEllipseScore);
#endif
	asrs.clear();

#ifdef DETAIL_BREAKDOWN
	t_cluster = (cv::getTickCount() - tmp_st) * 1000 / cv::getTickFrequency();
#endif


}
void FLED::run_FLED(Mat Img_G)
{
	// Pre-Processing
#ifdef DETAIL_BREAKDOWN
	double tmp_st;
	t_allfitting = t_allvalidation = 0;
	fitting_time = valitation_time = 0;
	tmp_st = cv::getTickCount();
#endif

	//vld_use_time = 0;
	double t1, t2;
	sum_time = 0; fitnum = 0;
	case_stat[0] = case_stat[1] = case_stat[2] = case_stat[3] = 0;
	//isBIDIR = 0;

	initFrame();

	iROWS = Img_G.rows; iCOLS = Img_G.cols;

	// ---------------- opencv版本Canny-----------
	//图像模糊的阈值需要自适应，这个应该与图像质量有关

	const double var = 0.6 / 0.8, sprec = 3;
	int L = int(var*sqrt(2 * sprec*log(10.0))) + 1;
	GaussianBlur(Img_G, Img_G, cv::Size(2 * L + 1, 2 * L + 1), var);
	//cv::medianBlur(Img_G, Img_G, 3);

	//Canny，两个参数已自适应
	int low, high;
	calCannyThreshold(Img_G, low, high);
	Canny(Img_G, imgCanny, low, high);

	//t1 = cv::getTickCount();
	uchar *_imgCanny = (uchar*)imgCanny.data;
	//边缘轮廓查找，个数小于T_edge_num(参数自适应)将会被扔掉
	findContours(_imgCanny);


	BoldEdge(_imgCanny, edgeContours); //将图像进行加粗，方便查找边缘点

#ifdef DETAIL_BREAKDOWN
	t_preprocess = (cv::getTickCount() - tmp_st) * 1000 / cv::getTickFrequency();

	tmp_st = cv::getTickCount();
#endif

	//t1 = cv::getTickCount();
	const int edgeContoursNum = int(edgeContours.size());
	for (int i = 0; i < edgeContoursNum; i++)
	{
#if ADAPT_APPROX_CONTOURS
		//adaptApproximateContours(edgeContours[i], oneContour);
		adaptApproxPolyDP(edgeContours[i], oneContour);
#else
		approxPolyDP(edgeContours[i], oneContour, _T_dp, false);
#endif
		if (oneContour.size() < MIN_DP_CONTOUR_NUM)
			continue;
		dpContours.push_back(oneContour);
	}
	//t2 = cv::getTickCount();
	//cout << "RDP using time:" << (t2 - t1) * 1000 / cv::getTickFrequency() << endl;

	//drawDPContours();


	const int dpContoursNum = int(dpContours.size());
	for (int i = 0; i < dpContoursNum; i++)
	{
		FSA_Segment(dpContours[i]);
	}

#ifdef DETAIL_BREAKDOWN
	t_arcsegment = (cv::getTickCount() - tmp_st) * 1000 / cv::getTickFrequency();
	tmp_st = cv::getTickCount();
#endif

	SortArcs(FSA_ArcContours); // 针对弧段个数进行排序

	CreateArcSearchRegion(FSA_ArcContours); //计算所有弧段的搜索区间

	int fsa_Arcs = int(FSA_ArcContours.size());
	if (fsa_Arcs == 0)
		return;
	getArcs_KDTrees(FSA_ArcContours);


	Arcs_Grouping(FSA_ArcContours);

	//writeLinkMatrix();
	//t2 = cv::getTickCount();
	//sum_time += (t2 - t1) * 1000 / cv::getTickFrequency();

	const char *_linkMatrix = LinkMatrix.GetDataPoint();
	getlinkArcs(_linkMatrix, fsa_Arcs);

	//drawFSA_ArcContours();
	//cv::waitKey(1);


	visited.Update(fsa_Arcs);
	searched.Update(fsa_Arcs);



	GetArcMat(ArcFitMat, FSA_ArcContours); //计算每个弧段的拟合矩阵


	_arc_grouped_label.clear();
	_arc_grouped_label.resize(fsa_Arcs, 0);


	for (int root_idx = 0; root_idx < fsa_Arcs; root_idx++)
	{
		if (FSA_ArcContours[root_idx].size() <= 3)
			continue;

		if (_arc_grouped_label[root_idx] != 0)
			continue;
		//t1 = cv::getTickCount();
		search_group[0].clear(), search_group[1].clear();//清空上一租数据点
		search_arcMats[0].clear(), search_arcMats[1].clear(); //清空上一组矩阵拟合数据点
		searched.clean();

		temp.push_back(root_idx);//加入根节点弧段
		*visited[root_idx] = 1; //当前节点访问过
		memcpy(fitArcTemp.val, ArcFitMat[root_idx].val, sizeof(double)*MAT_NUMBER);

		//t1 = cv::getTickCount();
		PosteriorArcsSearch2(root_idx);
		//t2 = (double)cv::getTickCount();



		temp.push_back(root_idx);
		*visited[root_idx] = 1;
		AnteriorArcsSearch(root_idx);

		//t1 = cv::getTickCount();
		sortCombine(fitComb_LR, search_arcMats, search_group);
		//t2 = (double)cv::getTickCount();
		//printf("The %dth root arc: the search_group size is %d, %d\n", root_idx, search_group[0].size(), search_group[1].size());
		//t1 = cv::getTickCount();
		BiDirectionVerification(fitComb_LR, search_arcMats, search_group, _arc_grouped_label);
		//t2 = (double)cv::getTickCount();
		//sum_time += (t2 - t1) * 1000 / cv::getTickFrequency();
	}

#ifdef DETAIL_BREAKDOWN
	t_allgrouping = (cv::getTickCount() - tmp_st) * 1000 / cv::getTickFrequency();
	tmp_st = cv::getTickCount();
#endif


#if SELECT_CLUSTER_METHOD == OUR_CLUSTER_METHOD
	EllipseNonMaximumSuppression(detEllipses, detEllipseScore, 0.7);
#elif SELECT_CLUSTER_METHOD == PRASAD_CLUSTER_METHOD
	ClusterEllipses(detEllipses, detEllipseScore);
#endif
	asrs.clear();
	//cout << vld_use_time << endl;


#ifdef DETAIL_BREAKDOWN
	t_cluster = (cv::getTickCount() - tmp_st) * 1000 / cv::getTickFrequency();
#endif
}





void FLED::run_FLED_MultiScale(Mat Img_G)
{
	if (!_multiScaleFpnConfig.enable)
	{
		run_FLED(Img_G);
		return;
	}

	const double baselineTheta = _theta_fsa;
	const double baselineLambda = _length_fsa;
	const double baselineValidationThreshold = _T_val;

	run_FLED(Img_G);
	const vector<cv::RotatedRect> baselineEllipses = detEllipses;
	const vector<double> baselineScores = detEllipseScore;

	vector<MultiScaleCandidate> candidates;
	for (int scale : _multiScaleFpnConfig.scales)
	{
		if (scale <= 1)
			continue;

		for (double validationThreshold : _multiScaleFpnConfig.validationThresholds)
		{
			cv::Mat upscaled;
			const int interpolation = _multiScaleFpnConfig.useCubicInterpolation ? cv::INTER_CUBIC : cv::INTER_LINEAR;
			cv::resize(
				Img_G,
				upscaled,
				cv::Size(Img_G.cols * scale, Img_G.rows * scale),
				0.0,
				0.0,
				interpolation);

			SetParameters(baselineTheta, baselineLambda, validationThreshold);
			run_FLED(upscaled);
			remapMultiScaleCandidates(scale, validationThreshold, detEllipses, detEllipseScore, candidates);
		}
	}

	SetParameters(baselineTheta, baselineLambda, baselineValidationThreshold);
	detEllipses = baselineEllipses;
	detEllipseScore = baselineScores;
	fuseMultiScaleCandidates(baselineEllipses, baselineScores, candidates);
}

void FLED::remapMultiScaleCandidates(
	int scale,
	double validationThreshold,
	const vector<cv::RotatedRect> &detections,
	const vector<double> &scores,
	vector<MultiScaleCandidate> &outCandidates) const
{
	const double invScale = 1.0 / double(scale);
	for (size_t idx = 0; idx < detections.size(); ++idx)
	{
		MultiScaleCandidate candidate;
		candidate.scale = scale;
		candidate.validationThreshold = validationThreshold;
		candidate.branchLabel = "s" + std::to_string(scale) + "_t" + std::to_string(int(std::lround(validationThreshold * 100.0)));
		candidate.ellipse = detections[idx];
		candidate.ellipse.center.x *= invScale;
		candidate.ellipse.center.y *= invScale;
		candidate.ellipse.size.width *= invScale;
		candidate.ellipse.size.height *= invScale;
		candidate.score = idx < scores.size() ? scores[idx] : 0.0;

		const double semiMajor = candidate.ellipse.size.width * 0.5;
		const double semiMinor = candidate.ellipse.size.height * 0.5;
		const double remappedRadius = std::sqrt(std::max(0.0, semiMajor * semiMinor));
		if (remappedRadius <= _multiScaleFpnConfig.remapRadiusMax)
		{
			outCandidates.push_back(candidate);
		}
	}
}

void FLED::buildMultiScaleIoUGraph(
	const vector<MultiScaleCandidate> &candidates,
	vector<vector<int>> &graph) const
{
	const int candidateNum = int(candidates.size());
	graph.assign(candidateNum, vector<int>());
	for (int i = 0; i < candidateNum; ++i)
	{
		for (int j = i + 1; j < candidateNum; ++j)
		{
			cv::RotatedRect lhs = candidates[i].ellipse;
			cv::RotatedRect rhs = candidates[j].ellipse;
			if (EllipseOverlap(lhs, rhs) >= _multiScaleFpnConfig.fusionIoU)
			{
				graph[i].push_back(j);
				graph[j].push_back(i);
			}
		}
	}
}

void FLED::findMultiScaleClusters(
	const vector<vector<int>> &graph,
	vector<vector<int>> &clusters) const
{
	const int nodeNum = int(graph.size());
	vector<unsigned char> visited(nodeNum, 0);
	clusters.clear();
	for (int root = 0; root < nodeNum; ++root)
	{
		if (visited[root] != 0)
			continue;

		vector<int> component;
		std::vector<int> queue(1, root);
		visited[root] = 1;
		size_t head = 0;
		while (head < queue.size())
		{
			const int current = queue[head++];
			component.push_back(current);
			for (int next : graph[current])
			{
				if (visited[next] == 0)
				{
					visited[next] = 1;
					queue.push_back(next);
				}
			}
		}
		clusters.push_back(component);
	}
}

int FLED::selectMultiScaleRepresentative(
	const vector<int> &cluster,
	const vector<MultiScaleCandidate> &candidates) const
{
	if (cluster.empty())
		return -1;

	int bestIdx = cluster.front();
	double bestAvgIoU = -1.0;
	double bestScore = -1.0;
	for (int idx : cluster)
	{
		double iouSum = 0.0;
		int iouCount = 0;
		for (int other : cluster)
		{
			if (other == idx)
				continue;
			cv::RotatedRect lhs = candidates[idx].ellipse;
			cv::RotatedRect rhs = candidates[other].ellipse;
			iouSum += EllipseOverlap(lhs, rhs);
			++iouCount;
		}
		const double avgIoU = iouCount > 0 ? iouSum / iouCount : 1.0;
		const double candidateScore = candidates[idx].score;
		if (avgIoU > bestAvgIoU ||
			(std::abs(avgIoU - bestAvgIoU) < 1e-9 && candidates[idx].scale > candidates[bestIdx].scale) ||
			(std::abs(avgIoU - bestAvgIoU) < 1e-9 && candidates[idx].scale == candidates[bestIdx].scale && candidateScore > bestScore))
		{
			bestIdx = idx;
			bestAvgIoU = avgIoU;
			bestScore = candidateScore;
		}
	}
	return bestIdx;
}

void FLED::fuseMultiScaleCandidates(
	const vector<cv::RotatedRect> &baselineEllipses,
	const vector<double> &baselineScores,
	const vector<MultiScaleCandidate> &candidates)
{
	if (candidates.empty())
	{
		detEllipses = baselineEllipses;
		detEllipseScore = baselineScores;
		return;
	}

	vector<vector<int>> graph;
	buildMultiScaleIoUGraph(candidates, graph);

	vector<vector<int>> clusters;
	findMultiScaleClusters(graph, clusters);

	detEllipses = baselineEllipses;
	detEllipseScore = baselineScores;

	for (const vector<int> &cluster : clusters)
	{
		vector<std::string> branches;
		bool hasScale2 = false;
		bool hasScale3 = false;
		for (int idx : cluster)
		{
			if (std::find(branches.begin(), branches.end(), candidates[idx].branchLabel) == branches.end())
				branches.push_back(candidates[idx].branchLabel);
			hasScale2 = hasScale2 || candidates[idx].scale == 2;
			hasScale3 = hasScale3 || candidates[idx].scale == 3;
		}

		if (int(branches.size()) < _multiScaleFpnConfig.minBranches)
			continue;
		if (_multiScaleFpnConfig.requireCrossScale && !(hasScale2 && hasScale3))
			continue;

		const int representative = selectMultiScaleRepresentative(cluster, candidates);
		if (representative < 0)
			continue;

		cv::RotatedRect selected = candidates[representative].ellipse;
		bool isDuplicate = false;
		for (const cv::RotatedRect &existing : detEllipses)
		{
			cv::RotatedRect lhs = selected;
			cv::RotatedRect rhs = existing;
			if (EllipseOverlap(lhs, rhs) >= _multiScaleFpnConfig.fusionIoU)
			{
				isDuplicate = true;
				break;
			}
		}
		if (!isDuplicate)
		{
			detEllipses.push_back(selected);
			detEllipseScore.push_back(candidates[representative].score);
		}
	}
}

void FLED::getArcs_KDTrees(vector< vector<Point> > & arcs)
{
	source_arcs.create(arcs.size(), 2, CV_32FC1);
	Point *temp;
	float *_source_arcs = (float*)source_arcs.data;
	int source_num = arcs.size();
	for (int i = 0; i < source_num; i++)
	{
		temp = &arcs[i][0];
		_source_arcs[i << 1] = temp->x;
		_source_arcs[(i << 1) + 1] = temp->y;
	}
	kdTree_Arcs.build(source_arcs, indexParams, cvflann::FLANN_DIST_L1);
}

void FLED::Arcs_Grouping(vector< vector<Point> > &fsaarcs)
{

	//cv::flann::KDTreeIndexParams indexParamst;
	//cv::flann::Index kdTree_Arcst;
	//kdTree_Arcst.build(source_arcs, indexParamst, cvflann::FLANN_DIST_L1);
	//Mat indices_arcst, dist_arcst;


	int fsaarcs_num = fsaarcs.size();
	int sub_arcnum, dist_l, searchNum, findIdx, linkval[2], group_res, _linkIdx;
	//const int maxArcs = std::min(32, fsaarcs_num);
	const int maxArcs = fsaarcs_num; // std::min(32, fsaarcs_num);
	int *_indices_arcs = NULL;
	float *_dist_arcs = NULL;
	Point *l1[4] = { NULL,NULL,NULL,NULL }, *l2[4] = { NULL,NULL,NULL,NULL };


	Point l1_l2_O, vecl[2], l1_l2[3], arc_n[3];

	unsigned char distType = 0x00;
	Point vecFeature[8];

	LinkMatrix.Update(fsaarcs_num*fsaarcs_num);// 更新邻接矩阵个数
	LinkWeight.Update(fsaarcs_num*fsaarcs_num);
	char* _LinkMatrix = LinkMatrix.GetDataPoint();
	float* _linkWeight = LinkWeight.GetDataPoint();

	int T_ij, T_ji;
	Point *_vec_data_i = NULL, *_vec_data_j = NULL; //获取vector的数据指针


	// 1-25 Add
	ArcSearchRegion asr1, asr2;
	ArcSearchRegion *_asrs = asrs.data(), *_asr1(NULL), *_asr2(NULL);

	for (int i = 0; i < fsaarcs_num; i++)
	{
		sub_arcnum = int(fsaarcs[i].size());
		_vec_data_i = fsaarcs[i].data();

		l1[0] = _vec_data_i;                   // A^{i}_1
		l1[1] = _vec_data_i + 1;               // A^{i}_2
		l1[2] = _vec_data_i + sub_arcnum - 2;  // A^{i}_{-2}
		l1[3] = _vec_data_i + sub_arcnum - 1;  // A^{i}_{-1}

		//asr1.create(l1[0], l1[1], l1[2], l1[3]); // Create Search Region, Modification of Prasad
		_asr1 = _asrs + i;
		vecFeature[0].x = l1[1]->x - l1[0]->x; vecFeature[0].y = l1[1]->y - l1[0]->y; //A0→A1
		vecFeature[1].x = l1[3]->x - l1[2]->x; vecFeature[1].y = l1[3]->y - l1[2]->y; //Aend-1Aend
		vecFeature[2].x = l1[3]->x - l1[0]->x; vecFeature[2].y = l1[3]->y - l1[0]->y;

		dist_l = abs(l1[0]->x - l1[3]->x) + abs(l1[0]->y - l1[3]->y); //KD-Tree Search Radius.
		query_arcs[0] = l1[3]->x; query_arcs[1] = l1[3]->y; // KD-Tree Search root.
		//利用kd树，搜索满足距离约束的点
		searchNum = kdTree_Arcs.radiusSearch(query_arcs, indices_arcs, dist_arcs, dist_l, maxArcs);
		//if (i == 0)
		//{
		//	cout << "Begin Error" << endl;
		//	cout << query_arcs[0] << "," << query_arcs[1] << endl;
		//	cout << indices_arcs << endl;
		//	cout << dist_arcs << endl;
		//	cout << dist_l << endl;
		//	cout << maxArcs << endl;

		//	float *_source_arcs = (float*)source_arcs.data;
		//	if (source_arcs.rows > 13)
		//	{
		//		cout << _source_arcs[13 * 2] << " , " << _source_arcs[13 * 2 + 1] << endl;
		//		//system("pause");
		//	}
		//	//kdTree_Arcst.radiusSearch(query_arcs, indices_arcst, dist_arcst, dist_l, maxArcs);
		//	//cout << "New KDTree" << endl;
		//	//cout << indices_arcst << endl;
		//	//cout << dist_arcst << endl;
		//	
		//}

		_indices_arcs = (int*)indices_arcs.data;
		_dist_arcs = (float*)dist_arcs.data;

		for (int j = 0; j < searchNum; j++)
		{
			distType = FLED_GROUPING_IBmA1_IAnB1;
			findIdx = _indices_arcs[j];// 找到的弧段编号
			_linkIdx = i*fsaarcs_num + findIdx; //获取邻接矩阵的编号
			if (_LinkMatrix[_linkIdx] != 0)
				continue;
			int findArcNum = fsaarcs[findIdx].size();
			_vec_data_j = fsaarcs[findIdx].data();   //获取被搜索的弧段指针

			l2[0] = _vec_data_j;                   // A^{j}_1
			l2[1] = _vec_data_j + 1;               // A^{j}_2
			l2[2] = _vec_data_j + findArcNum - 2;  // A^{j}_{-2}
			l2[3] = _vec_data_j + findArcNum - 1;  // A^{j}_{-1}

			//asr2.create(l2[0], l2[1], l2[2], l2[3]); // Create Search Region, Modification of Prasad
			_asr2 = _asrs + findIdx;
			vecFeature[3].x = l2[0]->x - l1[3]->x; vecFeature[3].y = l2[0]->y - l1[3]->y;
			vecFeature[7].x = l1[0]->x - l2[3]->x; vecFeature[7].y = l1[0]->y - l2[3]->y;

			T_ij = std::min(abs(vecFeature[1].x) + abs(vecFeature[1].y), // |A^{i}_{-1} - A^{i}_{-2}|
				abs(l2[0]->x - l2[1]->x) + abs(l2[0]->y - l2[1]->y));    // |A^{k}_1 - A^{k}_2|
			T_ji = std::min(abs(vecFeature[0].x) + abs(vecFeature[0].x), // |A^{i}_1 - A^{i}_2|
				abs(l2[2]->x - l2[3]->x) + abs(l2[2]->y - l2[3]->y));    // |A^{k}_{-1} - A^{k}_{-2}|
			if (_dist_arcs[j] < T_ij)
				distType = distType | FLED_GROUPING_CAnB1;
			if (abs(vecFeature[7].x) + abs(vecFeature[7].y) < T_ji)
				distType = distType | FLED_GROUPING_CBmA1;

			switch (distType)
			{
			case FLED_GROUPING_FBmA1_FAnB1:// 搜索点与尾点宽度都较远
			{
				//group_res = CASE_1(&asr1, &asr2);
				group_res = CASE_1(_asr1, _asr2);
				//case_stat[0]++;
				//group_res = Group4FAnB1_FBmA1(vecFeature, l1, l2);
				//group_res = Group4FAnB1_FBmA1(vecFeature, l1, l2, &asr1, &asr2);
				break;
			}
			case FLED_GROUPING_FBmA1_CAnB1:// 搜索点较近，尾点较远
			{
				group_res = Group4CAnB1_FBmA1(vecFeature, l1, l2);
				//case_stat[1]++;
				break;
			}
			case FLED_GROUPING_CBmA1_FAnB1:// 搜索点较远，尾点较近
			{
				group_res = Group4FAnB1_CBmA1(vecFeature, l1, l2);
				//case_stat[2]++;
				break;
			}
			case FLED_GROUPING_CBmA1_CAnB1: // 搜索点与尾点宽度都较近
			{
				if (findIdx == i)
				{
					group_res = 1;
					for (int j = 0; j < fsaarcs_num; j++)
					{
						_LinkMatrix[i*fsaarcs_num + j] = _LinkMatrix[j*fsaarcs_num + i] = -1;
					}
				}
				else
					group_res = Group4CAnB1_CBmA1(vecFeature, l1, l2);

				//case_stat[3]++;

				break;
			}
			default:
				break;
			}
			if (_weightedArcConfig.enable)
				_linkWeight[_linkIdx] = computeSoftLinkWeight(i, findIdx, l1, l2, _asr1, _asr2, int(_dist_arcs[j]));
			else
				_linkWeight[_linkIdx] = 0.0f;
			//if (i == 25 && findIdx == 32)
			//	cout << group_res << endl;
			//if (i == 32 && findIdx == 25)
			//	cout << group_res << endl;
			if (findIdx == i&&group_res == -1) //若弧段i不能跟自己相连，则这个弧段不可能跟其余的相连
			{
				continue;
			}


			_LinkMatrix[_linkIdx] = group_res;
			if (group_res == -1)
			{
				_LinkMatrix[findIdx*fsaarcs_num + i] = -1;
				_linkWeight[_linkIdx] = 0.0f;
			}
			//if (findIdx == i&&group_res==-1) //若弧段i不能跟自己相连，则这个弧段不可能跟其余的相连
			//{
			//	for (int j = 0; j < fsaarcs_num; j++)
			//	{
			//		_LinkMatrix[i*fsaarcs_num + j] = _LinkMatrix[j*fsaarcs_num + i] = -1;
			//	}
			//}


		}
	}
}

void FLED::getlinkArcs(const char *_linkMatrix, int arc_num)
{
	lA.resize(arc_num);
	const float *_linkWeight = LinkWeight.GetDataPoint();
	//查找第i个弧段相连的弧段和不相连的
	for (int i = 0; i < arc_num; i++)
	{
		//if (i == 25)
		//	cout << "Begin Error" << endl;
		int idx = i*arc_num;
		lA[i].clear();
		vector<int> linkingSoft;
		vector<int> linkedSoft;
		for (int j = 0; j < arc_num; j++)
		{
			//if (i == 32)
			//	cout << "Begin ERROR" << endl;
			//if (i == 25 && j == 32)
			//	cout << int(_linkMatrix[idx + j]) << endl;
			if (_linkMatrix[idx + j] == 1) // i 与j相连，则连接
				lA[i].idx_linking.push_back(j);
			else if (_linkMatrix[idx + j] == -1) //i与j不连，则不连
				lA[i].idx_notlink.push_back(j);
			if (_weightedArcConfig.enable &&
				_linkMatrix[idx + j] == 0 &&
				_linkWeight[idx + j] >= _weightedArcConfig.softLinkThreshold)
			{
				linkingSoft.push_back(j);
			}
			if (_linkMatrix[j*arc_num + i] == 1)
				lA[i].idx_linked.push_back(j);
			if (_weightedArcConfig.enable &&
				_linkMatrix[j*arc_num + i] == 0 &&
				_linkWeight[j*arc_num + i] >= _weightedArcConfig.softLinkThreshold)
			{
				linkedSoft.push_back(j);
			}
		}

		if (_weightedArcConfig.enable)
		{
			auto sortByWeight = [_linkWeight, arc_num, i](vector<int> &indices, bool reverseDirection)
			{
				std::sort(indices.begin(), indices.end(),
					[_linkWeight, arc_num, i, reverseDirection](int lhs, int rhs)
				{
					const float lhsWeight = reverseDirection ? _linkWeight[lhs * arc_num + i] : _linkWeight[i * arc_num + lhs];
					const float rhsWeight = reverseDirection ? _linkWeight[rhs * arc_num + i] : _linkWeight[i * arc_num + rhs];
					return lhsWeight > rhsWeight;
				});
			};

			sortByWeight(lA[i].idx_linking, false);
			sortByWeight(linkingSoft, false);
			sortByWeight(lA[i].idx_linked, true);
			sortByWeight(linkedSoft, true);

			if ((int)linkingSoft.size() > _weightedArcConfig.maxSoftNeighborsPerDirection)
				linkingSoft.resize(_weightedArcConfig.maxSoftNeighborsPerDirection);
			if ((int)linkedSoft.size() > _weightedArcConfig.maxSoftNeighborsPerDirection)
				linkedSoft.resize(_weightedArcConfig.maxSoftNeighborsPerDirection);

			lA[i].idx_linking.insert(lA[i].idx_linking.end(), linkingSoft.begin(), linkingSoft.end());
			lA[i].idx_linked.insert(lA[i].idx_linked.end(), linkedSoft.begin(), linkedSoft.end());
		}
	}
}

float FLED::getPairLinkWeight(int fromIdx, int toIdx) const
{
	if (!_weightedArcConfig.enable || fromIdx < 0 || toIdx < 0)
		return 0.0f;
	const int arcNum = int(FSA_ArcContours.size());
	if (fromIdx >= arcNum || toIdx >= arcNum)
		return 0.0f;
	return LinkWeight.GetDataPoint()[fromIdx * arcNum + toIdx];
}

float FLED::computeSoftLinkWeight(
	int srcIdx,
	int dstIdx,
	const Point * const *l1,
	const Point * const *l2,
	const ArcSearchRegion * const asri,
	const ArcSearchRegion * const asrk,
	int forwardGap) const
{
	if (!_weightedArcConfig.enable || srcIdx == dstIdx)
		return 0.0f;

	const float tailGap = float(abs(l1[3]->x - l2[0]->x) + abs(l1[3]->y - l2[0]->y));
	const float reverseGap = float(abs(l1[0]->x - l2[3]->x) + abs(l1[0]->y - l2[3]->y));
	const float localScale = float(std::max(4.0, std::min(
		double(abs(l1[3]->x - l1[2]->x) + abs(l1[3]->y - l1[2]->y) +
			abs(l2[1]->x - l2[0]->x) + abs(l2[1]->y - l2[0]->y)),
		18.0)));

	const float distScore = std::max(0.0f, 1.0f - tailGap / (2.5f * localScale + 6.0f));
	const float reverseScore = std::max(0.0f, 1.0f - reverseGap / (2.5f * localScale + 6.0f));

	const cv::Point2f t1(float(l1[3]->x - l1[2]->x), float(l1[3]->y - l1[2]->y));
	const cv::Point2f t2(float(l2[1]->x - l2[0]->x), float(l2[1]->y - l2[0]->y));
	const float n1 = std::sqrt(t1.x * t1.x + t1.y * t1.y);
	const float n2 = std::sqrt(t2.x * t2.x + t2.y * t2.y);
	float tangentScore = 0.0f;
	if (n1 > 1e-4f && n2 > 1e-4f)
	{
		const float cosine = (t1.x * t2.x + t1.y * t2.y) / (n1 * n2);
		tangentScore = std::max(0.0f, 0.5f * (cosine + 1.0f));
	}

	const bool mutualRegion = RegionConstraint(asri, asrk) && RegionConstraint(asrk, asri);
	const float regionScore = mutualRegion ? 1.0f : 0.35f;

	const vector<Point> &srcArc = FSA_ArcContours[srcIdx];
	const vector<Point> &dstArc = FSA_ArcContours[dstIdx];
	const Point &srcSt = srcArc.front();
	const Point &srcEd = srcArc.back();
	const Point &dstSt = dstArc.front();
	const Point &dstEd = dstArc.back();
	const float srcSpan = float(std::abs(data[dIDX(srcEd.x, srcEd.y)].edgeID - data[dIDX(srcSt.x, srcSt.y)].edgeID));
	const float dstSpan = float(std::abs(data[dIDX(dstEd.x, dstEd.y)].edgeID - data[dIDX(dstSt.x, dstSt.y)].edgeID));
	const float supportNorm = float(std::max(6.0, _T_edge_num));
	const float srcSupport = std::min(1.0f, srcSpan / supportNorm);
	const float dstSupport = std::min(1.0f, dstSpan / supportNorm);
	const float supportScore = 0.5f * (srcSupport + dstSupport);

	const float kdScore = std::max(0.0f, 1.0f - float(forwardGap) / std::max(8.0f, 2.0f * localScale));
	const float score =
		0.28f * distScore +
		0.18f * reverseScore +
		0.22f * tangentScore +
		0.17f * supportScore +
		0.15f * kdScore;

	return std::max(0.0f, std::min(1.0f, score * regionScore));
}

double FLED::computeGroupCompatibility(const vector<int> &group) const
{
	if (!_weightedArcConfig.enable || group.size() <= 1)
		return 0.0;

	double weightSum = 0.0;
	int weightCount = 0;
	for (size_t i = 0; i + 1 < group.size(); ++i)
	{
		const float forward = getPairLinkWeight(group[i], group[i + 1]);
		const float backward = getPairLinkWeight(group[i + 1], group[i]);
		const float pairWeight = std::max(forward, backward);
		if (pairWeight > 0.0f)
		{
			weightSum += pairWeight;
			++weightCount;
		}
	}

	return weightCount > 0 ? weightSum / weightCount : 0.0;
}

double FLED::computeCrossGroupCompatibility(const vector<int> &leftGroup, const vector<int> &rightGroup) const
{
	if (!_weightedArcConfig.enable || leftGroup.empty() || rightGroup.empty())
		return 0.0;

	double weightSum = 0.0;
	int weightCount = 0;
	for (int leftArc : leftGroup)
	{
		for (int rightArc : rightGroup)
		{
			const float lr = getPairLinkWeight(leftArc, rightArc);
			const float rl = getPairLinkWeight(rightArc, leftArc);
			const float pairWeight = std::max(lr, rl);
			if (pairWeight > 0.0f)
			{
				weightSum += pairWeight;
				++weightCount;
			}
		}
	}

	return weightCount > 0 ? weightSum / weightCount : 0.0;
}






bool FLED::FittingConstraint(double *_linkingMat, double *_linkedMat, cv::RotatedRect &fitres)
{
#ifdef DETAIL_BREAKDOWN
	double tmp_st;
#endif
	double fitingMat[MAT_NUMBER];
	if (_linkingMat != NULL&&_linkedMat != NULL)
	{
		for (int i = 0; i < MAT_NUMBER; i++)
			fitingMat[i] = _linkingMat[i] + _linkedMat[i];
	}
	else if (_linkingMat != NULL)
	{
		for (int i = 0; i < MAT_NUMBER; i++)
			fitingMat[i] = _linkingMat[i];
	}
	else if (_linkedMat != NULL)
	{
		for (int i = 0; i < MAT_NUMBER; i++)
			fitingMat[i] = _linkedMat[i];
	}


	double ellipse_error;
	//return false;
	//double t1, t2;
	//t1 = cv::getTickCount();
	//ElliFit(fitingMat, ellipse_error, fitres);
#ifdef DETAIL_BREAKDOWN
	tmp_st = cv::getTickCount();
#endif
	fitEllipse(fitingMat, ellipse_error, fitres);
#ifdef DETAIL_BREAKDOWN
	t_allfitting += (cv::getTickCount() - tmp_st) * 1000 / cv::getTickFrequency();
	fitting_time++;
#endif
	//t2 = (double)cv::getTickCount();
	//vld_use_time += (t2 - t1) * 1000 / cv::getTickFrequency();
	//fitnum++;
	//sum_time += (t2 - t1) * 1000 / cv::getTickFrequency();

	// Some obviously cases: 无法拟合出椭圆，椭圆尺寸大于图像尺寸，圆心在图像外
	if (ellipse_error < 0)
		return false;
	if (fitres.center.x < 0 || fitres.center.x >= iROWS || fitres.center.y<0 || fitres.center.y>iCOLS)
		return false;
	if (fitres.size.height*fitres.size.width > iROWS*iCOLS)
		return false;



	return true;
}
void FLED::ElliFit(double *data, double &error, cv::RotatedRect &res)
{
	double rp[5];
	double a1p, a2p, a11p, a22p, C2, alpha[9];
	int dot_num = data[14];
	//Mat buf_fit_data(6, 6, CV_64FC1, _buf_fit_data);
	cv::Point2d cen_dot(data[12] / (2 * dot_num), data[13] / (2 * dot_num));//the center of the fitting data.

	double A[9], B[9], D[9], C[9], B_invD_Bt[9], lambda;
	cv::Mat mB(3, 3, CV_64FC1, B), mD(3, 3, CV_64FC1, D), mB_invD_Bt(3, 3, CV_64FC1, B_invD_Bt);
	A[0] = data[0] - 2 * cen_dot.x*data[3] + 6 * cen_dot.x*cen_dot.x*data[5] - 3 * dot_num*pow(cen_dot.x, 4);
	A[3] = A[1] = data[1] - 3 * cen_dot.x*data[4] + 3 * cen_dot.x*cen_dot.x*data[8] - cen_dot.y*data[3] + cen_dot.x*cen_dot.y * 6 * data[5] - 6 * dot_num*cen_dot.y*pow(cen_dot.x, 3);
	A[6] = A[2] = data[2] - cen_dot.y*data[4] + cen_dot.y*cen_dot.y*data[5] - cen_dot.x*data[7] / 2 + cen_dot.x*cen_dot.y*data[8] * 2 + cen_dot.x*cen_dot.x*data[11] - 3 * dot_num*pow(cen_dot.x*cen_dot.y, 2);
	B[0] = data[3] - 6 * cen_dot.x*data[5] + 4 * dot_num*pow(cen_dot.x, 3);
	B[1] = data[4] - cen_dot.x*data[8] * 2 - 2 * cen_dot.y*data[5] + 4 * dot_num*cen_dot.x*cen_dot.x*cen_dot.y;
	B[2] = data[5] - dot_num*cen_dot.x*cen_dot.x;

	A[4] = 4 * A[2];
	A[7] = A[5] = data[6] - 1.5 * cen_dot.y*data[7] + 3 * cen_dot.y*cen_dot.y*data[8] - cen_dot.x*data[10] + 6 * cen_dot.x*cen_dot.y*data[11] - 6 * dot_num*cen_dot.x*pow(cen_dot.y, 3);
	B[3] = 2 * B[1];
	B[4] = data[7] - 4 * cen_dot.y*data[8] - 4 * cen_dot.x*data[11] + 8 * dot_num*cen_dot.x*cen_dot.y*cen_dot.y;
	B[5] = data[8] - 2 * dot_num*cen_dot.x*cen_dot.y;

	A[8] = data[9] - 2 * cen_dot.y*data[10] + 6 * cen_dot.y*cen_dot.y*data[11] - 3 * dot_num*pow(cen_dot.y, 4);
	B[6] = B[4] / 2;
	B[7] = data[10] - 6 * cen_dot.y*data[11] + 4 * dot_num*pow(cen_dot.y, 3);
	B[8] = data[11] - dot_num*cen_dot.y*cen_dot.y;

	D[0] = 4 * B[2];
	D[3] = D[1] = 2 * B[5];
	D[2] = 0;

	D[4] = 4 * B[8];
	D[5] = D[6] = D[7] = 0;
	D[8] = dot_num;

	mB_invD_Bt = mB*mD.inv()*mB.t();
	double *_mB_invD_Bt = (double*)mB_invD_Bt.data;
	for (int i = 0; i < 9; i++)
		C[i] = A[i] - _mB_invD_Bt[i];

	double detC, temp, detC22;
	detC = C[0] * (C[4] * C[8] - C[5] * C[7]) - C[1] * (C[3] * C[8] - C[5] * C[6]) + C[2] * (C[3] * C[7] - C[4] * C[6]);
	temp = (C[0] * C[4] - C[1] * C[1]);
	if (abs(temp) < 1e-6)
	{
		error = -1;
		return;
	}
	lambda = detC / temp;
	error = lambda;

	detC22 = C[0] * C[4] - C[3] * C[1];
	if (abs(detC22) < 1e-6)
	{
		error = -1;
		return;
	}

	alpha[0] = -(C[2] * C[4] - C[5] * C[1]) / detC22;
	alpha[1] = -(C[0] * C[5] - C[3] * C[2]) / detC22;
	alpha[2] = 1;

	Mat invD_B = mD.inv()*mB.t();
	double *_invD_B = (double*)invD_B.data;
	alpha[3] = -(_invD_B[0] * alpha[0] + _invD_B[1] * alpha[1] + _invD_B[2] * alpha[2]);
	alpha[4] = -(_invD_B[3] * alpha[0] + _invD_B[4] * alpha[1] + _invD_B[5] * alpha[2]);
	alpha[5] = -(_invD_B[6] * alpha[0] + _invD_B[7] * alpha[1] + _invD_B[8] * alpha[2]);




	double *_dls_X = alpha;
	double dls_k = _dls_X[0] * _dls_X[2] - _dls_X[1] * _dls_X[1];
	for (int i = 0; i < 6; i++)
		_dls_X[i] /= sqrt(abs(dls_k));



	rp[0] = _dls_X[1] * _dls_X[4] - _dls_X[2] * _dls_X[3];
	rp[1] = _dls_X[1] * _dls_X[3] - _dls_X[0] * _dls_X[4];
	if (fabs(_dls_X[0] - _dls_X[2]) > 1e-10)
		rp[4] = atan(2 * _dls_X[1] / (_dls_X[0] - _dls_X[2])) / 2;
	else
	{
		if (_dls_X[1] > 0)
			rp[4] = CV_PI / 4;
		else
			rp[4] = -CV_PI / 4;
	}
	//至此拟合出来的参数信息是以左上角0,0位置为原点，row为x轴，col为y轴为基准的
	a1p = cos(rp[4])*_dls_X[3] + sin(rp[4])*_dls_X[4];
	a2p = -sin(rp[4])*_dls_X[3] + cos(rp[4])*_dls_X[4];
	a11p = _dls_X[0] + tan(rp[4])*_dls_X[1];
	a22p = _dls_X[2] - tan(rp[4])*_dls_X[1];
	C2 = a1p*a1p / a11p + a2p*a2p / a22p - _dls_X[5];
	double dls_temp1 = C2 / a11p, dls_temp2 = C2 / a22p, dls_temp;
	if (dls_temp1 > 0 && dls_temp2 > 0)
	{
		rp[2] = sqrt(dls_temp1);
		rp[3] = sqrt(dls_temp2);
		if (rp[2] < rp[3])
		{
			if (rp[4] >= 0)
				rp[4] -= CV_PI / 2;
			else
				rp[4] += CV_PI / 2;
			dls_temp = rp[2];
			rp[2] = rp[3];
			rp[3] = dls_temp;
		}
	}
	else
	{
		error = -2;
		return;
	}
	res.center.x = (rp[0] + cen_dot.x)*PIXEL_SCALE;
	res.center.y = (rp[1] + cen_dot.y)*PIXEL_SCALE;
	res.size.width = 2 * rp[2] * PIXEL_SCALE;
	res.size.height = 2 * rp[3] * PIXEL_SCALE;
	res.angle = rp[4] / CV_PI * 180;



}
void FLED::fitEllipse(double * data, double & error, cv::RotatedRect &res)
{
	double rp[5], _buf_fit_data[36];
	double a1p, a2p, a11p, a22p, C2;
	int dot_num = data[14];
	Mat buf_fit_data(6, 6, CV_64FC1, _buf_fit_data);
	cv::Point2d cen_dot(data[12] / (2 * dot_num), data[13] / (2 * dot_num));//the center of the fitting data.

	_buf_fit_data[0] = data[0] - 2 * cen_dot.x*data[3] + 6 * cen_dot.x*cen_dot.x*data[5] - 3 * dot_num*pow(cen_dot.x, 4);
	_buf_fit_data[1] = data[1] - 3 * cen_dot.x*data[4] + 3 * cen_dot.x*cen_dot.x*data[8] - cen_dot.y*data[3] + cen_dot.x*cen_dot.y * 6 * data[5] - 6 * dot_num*cen_dot.y*pow(cen_dot.x, 3);
	_buf_fit_data[2] = data[2] - cen_dot.y*data[4] + cen_dot.y*cen_dot.y*data[5] - cen_dot.x*data[7] / 2 + cen_dot.x*cen_dot.y*data[8] * 2 + cen_dot.x*cen_dot.x*data[11] - 3 * dot_num*pow(cen_dot.x*cen_dot.y, 2);
	_buf_fit_data[3] = data[3] - 6 * cen_dot.x*data[5] + 4 * dot_num*pow(cen_dot.x, 3);
	_buf_fit_data[4] = data[4] - cen_dot.x*data[8] * 2 - 2 * cen_dot.y*data[5] + 4 * dot_num*cen_dot.x*cen_dot.x*cen_dot.y;
	_buf_fit_data[5] = data[5] - dot_num*cen_dot.x*cen_dot.x;

	_buf_fit_data[1 * 6 + 1] = 4 * _buf_fit_data[2];
	_buf_fit_data[1 * 6 + 2] = data[6] - 1.5 * cen_dot.y*data[7] + 3 * cen_dot.y*cen_dot.y*data[8] - cen_dot.x*data[10] + 6 * cen_dot.x*cen_dot.y*data[11] - 6 * dot_num*cen_dot.x*pow(cen_dot.y, 3);
	_buf_fit_data[1 * 6 + 3] = 2 * _buf_fit_data[4];
	_buf_fit_data[1 * 6 + 4] = data[7] - 4 * cen_dot.y*data[8] - 4 * cen_dot.x*data[11] + 8 * dot_num*cen_dot.x*cen_dot.y*cen_dot.y;
	_buf_fit_data[1 * 6 + 5] = data[8] - 2 * dot_num*cen_dot.x*cen_dot.y;

	_buf_fit_data[2 * 6 + 2] = data[9] - 2 * cen_dot.y*data[10] + 6 * cen_dot.y*cen_dot.y*data[11] - 3 * dot_num*pow(cen_dot.y, 4);
	_buf_fit_data[2 * 6 + 3] = _buf_fit_data[1 * 6 + 4] / 2;
	_buf_fit_data[2 * 6 + 4] = data[10] - 6 * cen_dot.y*data[11] + 4 * dot_num*pow(cen_dot.y, 3);
	_buf_fit_data[2 * 6 + 5] = data[11] - dot_num*cen_dot.y*cen_dot.y;

	_buf_fit_data[3 * 6 + 3] = 4 * _buf_fit_data[5];
	_buf_fit_data[3 * 6 + 4] = 2 * _buf_fit_data[1 * 6 + 5];
	_buf_fit_data[3 * 6 + 5] = 0;

	_buf_fit_data[4 * 6 + 4] = 4 * _buf_fit_data[2 * 6 + 5];
	_buf_fit_data[4 * 6 + 5] = 0;

	_buf_fit_data[5 * 6 + 5] = dot_num;
	for (int i = 0; i < 6; i++)
	{
		for (int j = i + 1; j < 6; j++)
		{
			_buf_fit_data[j * 6 + i] = _buf_fit_data[i * 6 + j];
		}
	}
	double t1, t2;
	//t1 = cv::getTickCount();
	//ElliFit(data, error, res);
	//t2 = (double)cv::getTickCount();
	eigen(buf_fit_data, dls_D, dls_V);
	//cout << buf_fit_data << endl;

	//cout << "ElliFit Time:" << (t2 - t1) * 1000 / cv::getTickFrequency() << endl;
	//sum_time += (t2 - t1) * 1000 / cv::getTickFrequency();
	//cout << dls_D << '\n' << dls_V << endl;
	double *_dls_D = (double*)dls_D.data;
	if (_dls_D[5] > 1e-10)
	{
		for (int i = 0; i < 6; i++)
			dls_V.row(i) = dls_V.row(i) / sqrt(fabs(_dls_D[i]));
		buf_fit_data = dls_V*dls_C*dls_V.t();
		eigen(buf_fit_data, dls_D2, dls_V2);
		//		cout << dls_D2 << '\n' << dls_V2 << endl;
		double *_dls_D2 = (double*)dls_D2.data;
		if (_dls_D2[0] > 0)
		{
			error = PIXEL_SCALE*PIXEL_SCALE / _dls_D2[0] / dot_num;
			dls_X = dls_V2.row(0)*dls_V;
		}
		else
		{
			error = -1;
			return;
		}
	}
	else
		dls_X = dls_V.row(5);
	//cout << dls_X << endl;
	//system("pause");
	double *_dls_X = (double*)dls_X.data;
	double dls_k = _dls_X[0] * _dls_X[2] - _dls_X[1] * _dls_X[1];
	dls_X = dls_X / sqrt(abs(dls_k));



	rp[0] = _dls_X[1] * _dls_X[4] - _dls_X[2] * _dls_X[3];
	rp[1] = _dls_X[1] * _dls_X[3] - _dls_X[0] * _dls_X[4];
	if (fabs(_dls_X[0] - _dls_X[2]) > 1e-10)
		rp[4] = atan(2 * _dls_X[1] / (_dls_X[0] - _dls_X[2])) / 2;
	else
	{
		if (_dls_X[1] > 0)
			rp[4] = CV_PI / 4;
		else
			rp[4] = -CV_PI / 4;
	}
	//至此拟合出来的参数信息是以左上角0,0位置为原点，row为x轴，col为y轴为基准的
	a1p = cos(rp[4])*_dls_X[3] + sin(rp[4])*_dls_X[4];
	a2p = -sin(rp[4])*_dls_X[3] + cos(rp[4])*_dls_X[4];
	a11p = _dls_X[0] + tan(rp[4])*_dls_X[1];
	a22p = _dls_X[2] - tan(rp[4])*_dls_X[1];
	C2 = a1p*a1p / a11p + a2p*a2p / a22p - _dls_X[5];
	double dls_temp1 = C2 / a11p, dls_temp2 = C2 / a22p, dls_temp;
	if (dls_temp1 > 0 && dls_temp2 > 0)
	{
		rp[2] = sqrt(dls_temp1);
		rp[3] = sqrt(dls_temp2);
		if (rp[2] < rp[3])
		{
			if (rp[4] >= 0)
				rp[4] -= CV_PI / 2;
			else
				rp[4] += CV_PI / 2;
			dls_temp = rp[2];
			rp[2] = rp[3];
			rp[3] = dls_temp;
		}
	}
	else
	{
		error = -2;
		return;
	}
	res.center.x = (rp[0] + cen_dot.x)*PIXEL_SCALE;
	res.center.y = (rp[1] + cen_dot.y)*PIXEL_SCALE;
	res.size.width = 2 * rp[2] * PIXEL_SCALE;
	res.size.height = 2 * rp[3] * PIXEL_SCALE;
	res.angle = rp[4] / CV_PI * 180;
}




void FLED::calCannyThreshold(cv::Mat &ImgG, int &low, int &high)
{
	Mat ImgT, dx, dy, grad;
	cv::resize(ImgG, ImgT, cv::Size(ImgG.cols / 10, ImgG.rows / 10));
	cv::Sobel(ImgT, dx, CV_16SC1, 1, 0);
	cv::Sobel(ImgT, dy, CV_16SC1, 0, 1);
	short *_dx = (short*)dx.data, *_dy = (short*)dy.data;

	int subpixel_num = dx.rows*dx.cols;
	grad.create(1, subpixel_num, CV_32SC1);
	int* _grad = (int*)grad.data;
	int maxGrad(0);
	for (int i = 0; i < subpixel_num; i++)
	{
		_grad[i] = std::abs(_dx[i]) + std::abs(_dy[i]);
		if (maxGrad < _grad[i])
			maxGrad = _grad[i];
	}

	//set magic numbers
	const int NUM_BINS = 64;
	const double percent_of_pixels_not_edges = 0.7;
	const double threshold_ratio = 0.4;
	int bins[NUM_BINS] = { 0 };


	//compute histogram
#if defined(__GNUC__)
        int bin_size = std::floor(maxGrad / float(NUM_BINS) + 0.5f) + 1;
#else
        int bin_size = std::floorf(maxGrad / float(NUM_BINS) + 0.5f) + 1;
#endif
	if (bin_size < 1) bin_size  = 1;
	for (int i = 0; i < subpixel_num; i++)
	{
		bins[_grad[i] / bin_size]++;
	}

	//% Select the thresholds
	float total(0.f);
	float target = float(subpixel_num * percent_of_pixels_not_edges);

	high = 0;
	while (total < target)
	{
		total += bins[high];
		high++;
	}
	//	high *= bin_size;
	high *= (255.0f / NUM_BINS);
	//	low = std::min((int)std::floor(threshold_ratio * float(high)), 30);
	low = threshold_ratio*float(high);

}


void FLED::CreateArcSearchRegion(vector< vector<Point> > &fsaarcs)
{

	const int arc_num = fsaarcs.size();
	Point *_arc_data = NULL;
	int subarc_num;
	ArcSearchRegion asrTemp;

	//	asrs.resize(arc_num);


	for (int i = 0; i < arc_num; i++)
	{
		subarc_num = fsaarcs[i].size();
		_arc_data = fsaarcs[i].data();
		asrTemp.create(_arc_data, _arc_data + 1, _arc_data + subarc_num - 2, _arc_data + subarc_num - 1);
		asrs.push_back(asrTemp);
	}

}
