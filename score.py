import json
import pandas as pd
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from typing import Union, List
from IPython.display import display
import os

STRICT_FIELDS = ["rule_type", "reference", "depend_on"]
SEMANTIC_FIELDS = ["subject", "object", "test", "consequence","tag"]

class SentenceEmbeddingModel:
    def __init__(self, model_name, batch_size=16):
        from sentence_transformers import SentenceTransformer, util
        self.model = SentenceTransformer(model_name)
        self.util = util

    def calculate_similarity(self, sentence1, sentence2):
        emb1 = self.model.encode(sentence1, convert_to_tensor=True)
        emb2 = self.model.encode(sentence2, convert_to_tensor=True)
        score = self.util.cos_sim(emb1, emb2).item()
        return score

class Score:
    def __init__(self, 
                 pred_json_path: str = None, 
                 gt_json_path: str = None, 
                 pred_data: List[dict] = None, 
                 gt_data: List[dict] = None):
        self.pred_json_path = pred_json_path
        self.gt_json_path = gt_json_path

        if pred_data and gt_data:
            self.pred_data = pred_data
            self.gt_data = gt_data
        elif pred_json_path and gt_json_path:
            self.pred_data = self._load_json(pred_json_path)
            self.gt_data = self._load_json(gt_json_path)
        else:
            raise ValueError("Must provide either JSON paths or data lists.")

    def _load_json(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def calc_score(self):
        results = []
        total_samples = len(self.pred_data)
        total_rules = 0
        total_con_rules = 0
        num_con_rules = 0

        model = SentenceEmbeddingModel("all-MiniLM-L6-v2", batch_size=16)

        pred_rules = []
        gt_rules = []
        ext_rate = []

        for i in range(total_samples):
            _pred_rules = self.pred_data[i]
            _gt_rules = self.gt_data[i]

            if len(_pred_rules) == len(_gt_rules):
                num_con_rules += 1

            for j in range(min(len(_pred_rules), len(_gt_rules))):
                total_rules += 1
                if _pred_rules[j]['rule_id'] == _gt_rules[j]['rule_id']:
                    total_con_rules += 1
                else:
                    _pred_id = _pred_rules[j]['rule_id']
                    _gt_id = _gt_rules[j]['rule_id']
                    print(f'_pred_rules is {_pred_id}')
                    print(f'_gt_rules is {_gt_id}')
                pred_rules.append(_pred_rules[j])
                gt_rules.append(_gt_rules[j])
                ext_rate.append(min(1.0, len(_pred_rules)/len(_gt_rules)))

        strict_score = self.evaluate_strict_fields(pred_rules, gt_rules)
        semantic_scores = self.evaluate_semantic_fields(pred_rules, gt_rules, model=model)

        for i in range(len(gt_rules)):
            result_row = {
                'sample_idx': i // max(len(gt_rules), 1),
                'rule_idx': i
            }
            for field in STRICT_FIELDS:
                result_row[field] = int(pred_rules[i].get(field) == gt_rules[i].get(field))
            result_row.update(semantic_scores[i])
            results.append(result_row)

        df = pd.DataFrame(results)

        strict_avg = df[STRICT_FIELDS].mean().to_dict()
        semantic_avg = df[SEMANTIC_FIELDS].mean().to_dict()

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        pd.Series(strict_avg).plot(kind='bar', color='skyblue')
        plt.title('Strict Fields Score')
        plt.ylim(0, 1)
        plt.grid(axis='y')

        plt.subplot(1, 2, 2)
        pd.Series(semantic_avg).plot(kind='bar', color='lightgreen')
        plt.title('Semantic Fields Score')
        plt.ylim(0, 1)
        plt.grid(axis='y')

        plt.tight_layout()
        chart_path = "field_score_chart.png"
        plt.savefig(chart_path)
        plt.show()

        rule_level_num_con = num_con_rules / total_samples
        rule_level_ID_con = total_con_rules / total_rules
        avg_strict = sum(strict_avg.values()) / len(strict_avg)
        avg_semantic = sum(semantic_avg.values()) / len(semantic_avg)
        avg_score = (rule_level_num_con + rule_level_ID_con + 3 * avg_strict + 5 * avg_semantic) / 10

        print("="*40)
        print("规则级评估报告：")
        print(f'规则数量一致性={rule_level_num_con:.2%}')
        print(f'规则ID一致性={rule_level_ID_con:.2%}')
        print("="*40)
        print("字段级评估报告：")
        print("="*40)
        print("\n严格字段准确率：")
        display(pd.DataFrame({
            'Field': STRICT_FIELDS,
            'Accuracy': [strict_avg[f] for f in STRICT_FIELDS]
        }))

        print("\n语义字段平均分：")
        display(pd.DataFrame({
            'Field': SEMANTIC_FIELDS,
            'Score': [semantic_avg[f] for f in SEMANTIC_FIELDS]
        }))

        print("\n总体统计：")
        print(f"严格字段平均准确率：{avg_strict:.2%}")
        print(f"语义字段平均得分：{avg_semantic:.2%}")
        print(f"所有字段平均得分：{avg_score:.2%}")

        # ====================
        # 文件输出配置
        # ====================
        report_path = self.pred_json_path.split(".")[0] + "_evaluation_report.md"

        # ====================
        # 生成报告内容
        # ====================
        report_content = f"""
# 规则抽取评估报告

## 评估配置
- 预测文件：`{self.pred_json_path}`
- 真实数据：`{self.gt_json_path}`
- 评估时间：{pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}

## 规则级准确率：
- **规则数量一致性**：{rule_level_num_con:.2%}
- **规则ID一致性**：{rule_level_ID_con:.2%}

## 严格字段准确率
{df[STRICT_FIELDS].mean().to_markdown()}

## 语义字段得分
{df[SEMANTIC_FIELDS].mean().to_markdown()}

## 总体统计
- **严格字段平均准确率**：{avg_strict:.2%}
- **语义字段平均得分**：{avg_semantic:.2%}
- **综合得分（0.5*严格+0.5*语义）**：{avg_score:.2%}

![评估图表]({chart_path})
"""

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"评估报告已保存至：{report_path}")

    def evaluate_strict_fields(self, pred_item, gt_item, rate=1):
        match_count = 0
        total = len(gt_item)
        for g_rule in gt_item:
            for p_rule in pred_item:
                if all(p_rule.get(f) == g_rule.get(f) for f in STRICT_FIELDS):
                    match_count += 1
                    break
        return match_count / total * rate if total > 0 else 0

    def evaluate_semantic_fields(self, pred_item, gt_item, model=None, rate=1):
        if model is None:
            model = SentenceEmbeddingModel("all-MiniLM-L6-v2", batch_size=16)
        scores = []
        gt_rules = gt_item
        pred_rules = pred_item
        for g_rule in gt_rules:
            best_score = 0
            best_field_scores = {field: 0.0 for field in SEMANTIC_FIELDS}
            for p_rule in pred_rules:
                field_scores = {}
                total_score = 0
                for field in SEMANTIC_FIELDS:
                    pred_val = str(p_rule.get(field, ""))
                    gt_val = str(g_rule.get(field, ""))
                    sim = round(model.calculate_similarity(pred_val, gt_val), 2)
                    field_scores[field] = sim * rate
                    total_score += sim
                avg_score = total_score / len(SEMANTIC_FIELDS)
                if avg_score > best_score:
                    best_score = avg_score
                    best_field_scores = field_scores
            scores.append(best_field_scores)
        return scores

    def plot_stats(self):
        lens = [len(x['extracted_rules']) for x in self.pred_data]
        plt.hist(lens, bins=20)
        plt.title('Distribution of Extracted Rules Count')
        plt.xlabel('Rule Count')
        plt.ylabel('Frequency')
        plt.show()

# 示例用法：
# scorer = Score(pred_json_path='pred.json', gt_json_path='gt.json')
# scorer.calc_score()
# scorer.plot_stats()

