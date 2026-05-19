#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <cmath>

using namespace std;

// 一条关联规则
struct AssociationRule {
    set<string> antecedent;     // 前件
    set<string> consequent;     // 后件
    int supportCount;           // 前件和后件同时出现的次数
    double support;             // 支持度
    double confidence;          // 置信度
    double lift;                // 提升度
};

// 读取 Mushroom 数据集，一行记录当成一笔事务
vector<set<string>> loadMushroomData(const string& filename) {
    vector<set<string>> dataset;
    ifstream file(filename);
    string line;

    if (!file.is_open()) {
        cerr << "Error: 无法打开文件 " << filename << endl;
        return dataset;
    }

    while (getline(file, line)) {
        if (line.empty()) continue;

        stringstream ss(line);
        string value;
        set<string> transaction;
        int colIndex = 0;

        while (getline(ss, value, ',')) {
            // 加列号区分不同属性里的相同字母
            string item = "Col" + to_string(colIndex) + "=" + value;
            transaction.insert(item);
            colIndex++;
        }

        dataset.push_back(transaction);
    }

    file.close();
    return dataset;
}

string itemsetToString(const set<string>& itemset) {
    stringstream ss;
    ss << "{ ";
    for (auto it = itemset.begin(); it != itemset.end(); ++it) {
        ss << *it << (next(it) == itemset.end() ? "" : ", ");
    }
    ss << " }";
    return ss.str();
}

class Apriori {
private:
    double minSupportRatio;     // 最小支持度比例
    double minConfidenceRatio;  // 最小置信度比例
    int minSupportCount;        // 最小支持次数
    vector<set<string>> transactions;
    map<set<string>, int> supportCounts; // 保存算过的支持次数

    // 计算某个项集在数据集中出现了多少次
    int countSupport(const set<string>& itemset) const {
        int count = 0;
        for (const auto& transaction : transactions) {
            if (includes(transaction.begin(), transaction.end(), itemset.begin(), itemset.end())) {
                count++;
            }
        }
        return count;
    }

    // 支持度会反复用到，算过一次就先存起来
    int getSupportCount(const set<string>& itemset) {
        auto it = supportCounts.find(itemset);
        if (it != supportCounts.end()) {
            return it->second;
        }

        int count = countSupport(itemset);
        supportCounts[itemset] = count;
        return count;
    }

    // Apriori 剪枝：候选项集的所有子集都必须是频繁项集
    bool allSubsetsFrequent(const set<string>& candidate, const set<set<string>>& prevFrequentItemsets) const {
        for (const auto& item : candidate) {
            set<string> subset = candidate;
            subset.erase(item);
            if (prevFrequentItemsets.find(subset) == prevFrequentItemsets.end()) {
                return false;
            }
        }
        return true;
    }

    // 由 L(k-1) 连接生成候选项集 Ck
    set<set<string>> generateCandidates(const set<set<string>>& prevFrequentItemsets, int k) const {
        set<set<string>> candidates;
        vector<set<string>> prevItems(prevFrequentItemsets.begin(), prevFrequentItemsets.end());

        for (size_t i = 0; i < prevItems.size(); ++i) {
            for (size_t j = i + 1; j < prevItems.size(); ++j) {
                auto it1 = prevItems[i].begin();
                auto it2 = prevItems[j].begin();
                bool canJoin = true;

                // 前 k-2 个元素一样时才合并
                for (int m = 0; m < k - 2; ++m) {
                    if (*it1 != *it2) {
                        canJoin = false;
                        break;
                    }
                    it1++;
                    it2++;
                }

                if (canJoin) {
                    set<string> newItemset = prevItems[i];
                    newItemset.insert(*it2);

                    // 多做一步剪枝，减少后面扫描数据集的次数
                    if (static_cast<int>(newItemset.size()) == k &&
                        allSubsetsFrequent(newItemset, prevFrequentItemsets)) {
                        candidates.insert(newItemset);
                    }
                }
            }
        }

        return candidates;
    }

public:
    Apriori(double minSup, double minConf)
        : minSupportRatio(minSup), minConfidenceRatio(minConf), minSupportCount(0) {}

    // 挖掘所有满足最小支持度的频繁项集
    map<int, set<set<string>>> fit(const vector<set<string>>& dataset) {
        transactions = dataset;
        supportCounts.clear();
        minSupportCount = max(1, static_cast<int>(ceil(transactions.size() * minSupportRatio)));

        cout << "总记录数: " << transactions.size() << endl;
        cout << "最小支持度: " << fixed << setprecision(1) << minSupportRatio * 100
            << "%，最少出现 " << minSupportCount << " 次" << endl;
        cout << "最小置信度: " << fixed << setprecision(1) << minConfidenceRatio * 100 << "%" << endl;

        map<int, set<set<string>>> frequentItemsets;
        set<set<string>> currentL;

        cout << "\n--- 开始挖掘频繁项集 ---" << endl;

        // 第一次扫描数据集，先得到频繁 1-项集
        map<string, int> itemCounts;
        for (const auto& trans : transactions) {
            for (const auto& item : trans) {
                itemCounts[item]++;
            }
        }

        for (const auto& pair : itemCounts) {
            set<string> itemset = { pair.first };
            supportCounts[itemset] = pair.second;
            if (pair.second >= minSupportCount) {
                currentL.insert(itemset);
            }
        }

        if (!currentL.empty()) {
            frequentItemsets[1] = currentL;
            cout << "频繁 1-项集数量: " << currentL.size() << endl;
        }

        int k = 2;
        while (!currentL.empty()) {
            // 用上一轮的 L(k-1) 生成 Ck
            set<set<string>> candidates = generateCandidates(currentL, k);
            currentL.clear();

            // 再扫描数据集，留下支持度够高的候选项集
            for (const auto& candidate : candidates) {
                int support = getSupportCount(candidate);
                if (support >= minSupportCount) {
                    currentL.insert(candidate);
                }
            }

            if (!currentL.empty()) {
                frequentItemsets[k] = currentL;
                cout << "频繁 " << k << "-项集数量: " << currentL.size() << endl;
            }

            k++;
        }

        return frequentItemsets;
    }

    // 由频繁项集生成关联规则
    vector<AssociationRule> generateRules(const map<int, set<set<string>>>& frequentItemsets) {
        vector<AssociationRule> rules;
        int total = static_cast<int>(transactions.size());

        for (const auto& level : frequentItemsets) {
            if (level.first < 2) continue;

            for (const auto& itemset : level.second) {
                vector<string> items(itemset.begin(), itemset.end());
                size_t itemCount = items.size();
                size_t maskCount = static_cast<size_t>(1) << itemCount;
                int itemsetSupport = getSupportCount(itemset);

                // 用二进制 mask 枚举非空真子集，作为规则前件
                for (size_t mask = 1; mask < maskCount - 1; ++mask) {
                    set<string> antecedent;
                    set<string> consequent;

                    for (size_t i = 0; i < itemCount; ++i) {
                        if (mask & (static_cast<size_t>(1) << i)) {
                            antecedent.insert(items[i]);
                        }
                        else {
                            consequent.insert(items[i]);
                        }
                    }

                    int antecedentSupport = getSupportCount(antecedent);
                    int consequentSupport = getSupportCount(consequent);
                    double support = static_cast<double>(itemsetSupport) / total;
                    double confidence = static_cast<double>(itemsetSupport) / antecedentSupport;
                    double lift = confidence / (static_cast<double>(consequentSupport) / total);

                    // 这里只保留置信度达到要求的规则
                    if (confidence >= minConfidenceRatio) {
                        rules.push_back({ antecedent, consequent, itemsetSupport, support, confidence, lift });
                    }
                }
            }
        }

        // 先按置信度排，再按提升度和支持度排
        sort(rules.begin(), rules.end(), [](const AssociationRule& a, const AssociationRule& b) {
            if (a.confidence != b.confidence) return a.confidence > b.confidence;
            if (a.lift != b.lift) return a.lift > b.lift;
            return a.support > b.support;
        });

        return rules;
    }

    int storedSupportCount(const set<string>& itemset) const {
        auto it = supportCounts.find(itemset);
        if (it == supportCounts.end()) {
            return 0;
        }
        return it->second;
    }

    int transactionCount() const {
        return static_cast<int>(transactions.size());
    }
};

// 每一层只打印几个例子，避免输出太长
void printFrequentItemsets(const map<int, set<set<string>>>& result, const Apriori& model, int limit = 5) {
    cout << "\n--- 频繁项集示例 ---" << endl;

    for (const auto& level : result) {
        cout << "\n" << level.first << "-项集，共 " << level.second.size() << " 个" << endl;

        int printed = 0;
        for (const auto& itemset : level.second) {
            if (printed >= limit) {
                cout << "  ... 其余省略" << endl;
                break;
            }

            int count = model.storedSupportCount(itemset);
            double support = static_cast<double>(count) / model.transactionCount();
            cout << "  " << itemsetToString(itemset)
                << "  support=" << fixed << setprecision(4) << support
                << " (" << count << "次)" << endl;
            printed++;
        }
    }
}

// 打印置信度较高的规则
void printAssociationRules(const vector<AssociationRule>& rules, int limit = 20) {
    cout << "\n--- 关联规则结果 ---" << endl;
    cout << "共生成 " << rules.size() << " 条满足条件的规则，下面显示前 " << min(limit, static_cast<int>(rules.size())) << " 条" << endl;

    for (int i = 0; i < static_cast<int>(rules.size()) && i < limit; ++i) {
        const auto& rule = rules[i];
        cout << setw(2) << i + 1 << ". "
            << itemsetToString(rule.antecedent) << " => " << itemsetToString(rule.consequent)
            << "  support=" << fixed << setprecision(4) << rule.support
            << ", confidence=" << fixed << setprecision(4) << rule.confidence
            << ", lift=" << fixed << setprecision(4) << rule.lift
            << " (" << rule.supportCount << "次)" << endl;
    }
}

// 只看结论是蘑菇类别的规则
bool isClassRule(const AssociationRule& rule) {
    if (rule.consequent.size() != 1) {
        return false;
    }

    return rule.consequent.begin()->find("Col0=") == 0;
}

// 单独输出类别规则，报告里更好解释
void printClassRules(const vector<AssociationRule>& rules, int limit = 10) {
    cout << "\n--- 蘑菇类别相关规则 ---" << endl;
    cout << "Col0 是类别，Col0=e 表示可食用，Col0=p 表示有毒" << endl;

    int printed = 0;
    for (const auto& rule : rules) {
        if (!isClassRule(rule)) continue;

        cout << setw(2) << printed + 1 << ". "
            << itemsetToString(rule.antecedent) << " => " << itemsetToString(rule.consequent)
            << "  support=" << fixed << setprecision(4) << rule.support
            << ", confidence=" << fixed << setprecision(4) << rule.confidence
            << ", lift=" << fixed << setprecision(4) << rule.lift
            << " (" << rule.supportCount << "次)" << endl;

        printed++;
        if (printed >= limit) {
            break;
        }
    }

    if (printed == 0) {
        cout << "没有找到满足阈值的类别规则" << endl;
    }
}

int main() {
    cout << "--- 正在加载 UCI Mushroom 数据集 ---" << endl;
    vector<set<string>> dataset = loadMushroomData("mushroom/agaricus-lepiota.data");

    if (dataset.empty()) {
        return -1;
    }

    // 这里阈值设得高一点，先保证结果不会太多
    Apriori model(0.4, 0.8);
    map<int, set<set<string>>> result = model.fit(dataset);
    vector<AssociationRule> rules = model.generateRules(result);

    printFrequentItemsets(result, model);
    printAssociationRules(rules);
    printClassRules(rules);

    return 0;
}
