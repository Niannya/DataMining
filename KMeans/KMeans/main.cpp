#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <map>

using namespace std;

// 一条样本数据
struct Point {
    vector<double> features;    // 四个特征
    int clusterId;              // 聚到哪个簇，-1表示还未分配
    string trueLabel;           // 原来的类别

    Point(vector<double> f, string label = "") : features(f), clusterId(-1), trueLabel(label) {}
};

// 读取 Iris 数据集
vector<Point> loadIrisData(const string& filename) {
    vector<Point> dataset;
    ifstream file(filename);
    string line;

    if (!file.is_open()) {
        cerr << "Error: 无法打开文件 " << filename << "\n请确保 iris.data 和程序在同一目录下！" << endl;
        return dataset;
    }

    while (getline(file, line)) {
        if (line.empty()) continue;

        stringstream ss(line);
        string value;
        vector<double> features;
        string label;

        for (int i = 0; i < 4; ++i) {
            getline(ss, value, ',');
            features.push_back(stod(value));
        }
        getline(ss, label, ',');

        dataset.emplace_back(features, label);
    }
    file.close();
    return dataset;
}

// K-Means 算法
class KMeans {
private:
    int K;                      // 分成几簇
    int maxIterations;          // 最多迭代次数
    vector<vector<double>> centroids; // 每个簇的中心点

    double calculateDistance(const vector<double>& p1, const vector<double>& p2) const {
        double sum = 0.0;
        for (size_t i = 0; i < p1.size(); ++i) {
            sum += pow(p1[i] - p2[i], 2);
        }
        return sqrt(sum);
    }

public:
    KMeans(int k, int maxIter = 100) : K(k), maxIterations(maxIter) {}

    void fit(vector<Point>& points) {
        if (points.empty()) return;

        // 随机选 K 个样本当初始中心
        centroids.resize(K);
        vector<int> indices(points.size());
        iota(indices.begin(), indices.end(), 0);

        mt19937 g(42); // 固定种子，方便复现结果
        shuffle(indices.begin(), indices.end(), g);

        for (int i = 0; i < K; ++i) {
            centroids[i] = points[indices[i]].features;
        }

        // 不断分配样本并更新中心点
        for (int iter = 0; iter < maxIterations; ++iter) {
            // 把每个样本分到最近的中心
            for (auto& point : points) {
                double minDist = calculateDistance(point.features, centroids[0]);
                int closestCentroid = 0;
                for (int i = 1; i < K; ++i) {
                    double dist = calculateDistance(point.features, centroids[i]);
                    if (dist < minDist) {
                        minDist = dist;
                        closestCentroid = i;
                    }
                }
                point.clusterId = closestCentroid;
            }

            // 根据当前分簇重新计算中心
            vector<vector<double>> newCentroids(K, vector<double>(points[0].features.size(), 0.0));
            vector<int> counts(K, 0);
            for (const auto& point : points) {
                int clusterId = point.clusterId;
                for (size_t j = 0; j < point.features.size(); ++j) {
                    newCentroids[clusterId][j] += point.features[j];
                }
                counts[clusterId]++;
            }

            bool isChanged = false; // 中心点是否改变
            for (int i = 0; i < K; ++i) {
                if (counts[i] > 0) {
                    for (size_t j = 0; j < newCentroids[i].size(); ++j) {
                        newCentroids[i][j] /= counts[i];
                        // 移动不够大就当作没有变化
                        if (abs(centroids[i][j] - newCentroids[i][j]) > 1e-6) {
                            isChanged = true;
                        }
                    }
                    centroids[i] = newCentroids[i];
                }
            }

            // 中心点稳定后，不继续迭代
            if (!isChanged) {
                cout << "=> 算法在第 " << iter + 1 << " 次迭代后提前收敛！" << endl;
                break;
            }
        }
    }

    // 返回最后的中心点
    const vector<vector<double>>& getCentroids() const {
        return centroids;
    }

    // 计算 SSE
    double calculateSSE(const vector<Point>& points) const {
        double sse = 0.0;
        for (const auto& point : points) {
            if (point.clusterId != -1) {
                double dist = calculateDistance(point.features, centroids[point.clusterId]);
                sse += dist * dist;
            }
        }
        return sse;
    }
};

// 统计每个簇里面各类有多少
void printClusterStatistics(const vector<Point>& points, int K) {
    vector<int> clusterCounts(K, 0);
    vector<map<string, int>> labelCounts(K);

    for (const auto& point : points) {
        if (point.clusterId < 0 || point.clusterId >= K) continue;

        clusterCounts[point.clusterId]++;
        labelCounts[point.clusterId][point.trueLabel]++;
    }

    int majorityTotal = 0;

    cout << "\n各簇样本数量与真实类别分布:" << endl;
    for (int i = 0; i < K; ++i) {
        cout << "簇 " << i << "：样本数 = " << clusterCounts[i] << "，类别分布 = { ";

        int majorityCount = 0;
        bool first = true;
        for (const auto& labelCount : labelCounts[i]) {
            if (!first) {
                cout << ", ";
            }
            cout << labelCount.first << ": " << labelCount.second;
            majorityCount = max(majorityCount, labelCount.second);
            first = false;
        }

        cout << " }" << endl;
        majorityTotal += majorityCount;
    }

    double purity = points.empty() ? 0.0 : static_cast<double>(majorityTotal) / points.size();
    cout << "聚类纯度 : " << fixed << setprecision(4) << purity << endl;
}

int main() {
    cout << "--- 正在加载 UCI Iris 数据集 ---" << endl;
    vector<Point> dataset = loadIrisData("iris/iris.data");

    if (dataset.empty()) {
        return -1;
    }
    cout << "成功加载 " << dataset.size() << " 条数据。" << endl;

    // Iris 一共有 3 类
    int K = 3;
    KMeans model(K, 100);

    cout << "\n--- 开始训练 K-Means 模型 ---" << endl;
    model.fit(dataset);

    // 聚类效果
    double sse = model.calculateSSE(dataset);
    cout << "\n--- 聚类结果评估 ---" << endl;
    cout << "簇内误差平方和 : " << fixed << setprecision(4) << sse << endl;
    printClusterStatistics(dataset, K);

    // 输出中心点
    cout << "\n最终的聚类中心 :" << endl;
    const auto& finalCentroids = model.getCentroids();
    for (int i = 0; i < K; ++i) {
        cout << "簇 " << i << ": [ ";
        for (size_t j = 0; j < finalCentroids[i].size(); ++j) {
            cout << finalCentroids[i][j] << (j == finalCentroids[i].size() - 1 ? "" : ", ");
        }
        cout << " ]" << endl;
    }

    return 0;
}
