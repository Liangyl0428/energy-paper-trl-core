# 单篇论文 TRL 证据评估（核心算法版）

输入一篇论文的 **标题 + 摘要**，输出 Q/GDW 12566—2025 表 1 九级框架下的**证据支撑等级**、
可逐字回原文核对的引文、缺失属性、以及"为什么不是更高等级"。

本仓库只包含当前版本的核心算法（规则引擎 + 标准门禁 + 可选分类器适配器）。不包含训练数据、
评测集、历史权重与实验脚本。

```bash
python -m pip install -e '.[dev]'          # 核心：无第三方运行时依赖
python -m pip install -e '.[models]'       # 可选：SVM/SciBERT 适配器（sklearn + torch）
python -m pytest -q                        # 41 passed, 5 skipped（模型用例按需跳过）
```

> **安装方式必须是 `-e`（editable）**：标准层配置 `config/qgdw_12566_2025.json` 按仓库布局定位，
> 非 editable 安装（`pip install .`）不会携带它，`analyze()` 会抛 `FileNotFoundError`。
> 确实要装成 wheel 时请设置 `ENERGY_PAPER_TRL_STANDARD=/绝对路径/qgdw_12566_2025.json`。
>
> **不要与完整版同环境安装**：两者的发行名同为 `energy-paper-trl`、模块名同为 `energy_paper_trl`，
> 装进同一个虚拟环境会互相覆盖；请为本仓库单独建环境。

## 快速开始

命令行：

```bash
python -m energy_paper_trl \
  --title 'Grid-connected converter validation' \
  --abstract 'Laboratory experiments validated the proposed converter; test results met the grid-code target.'
```

Python：

```python
from energy_paper_trl import analyze

out = analyze(title, abstract)
print(out["level"], out["level_source"], out["status"])
print(out["standard_alignment"]["evidence_level"])
print(out["rationale"])
for item in out["evidence"]:
    print(item["quote"], item["start"], item["end"], item["truncated"])
```

## 算法逻辑

### 三层结构

```
标题+摘要
  ├─ 规则证据层  RuleTRLAnalyzer        → 关键词证据、逐句引文、规则等级
  ├─ 分类器层    SVM / SciBERT（可选）  → 模型等级、分数、句子归因（不是证据）
  └─ 标准门禁层  align_to_standard      → 逐属性 Gate、证据等级、缺失项
```

对外只暴露**一个**被采信等级，三层意见并列可查：

| 字段 | 含义 |
| --- | --- |
| `level` / `level_name` / `level_source` | 唯一采信等级。启用标准层时即证据等级（`level_source='evidence'`）；`standard=None` 时取规则等级或模型等级 |
| `levels` | `{evidence, rule, model}` 三值并列 |
| `status` | `public_evidence_preliminary` / `insufficient_evidence` / `not_applicable` / `evidence_conflict` |
| `rule_level` / `rule_status` | 规则层原始结论，便于对照 |
| `standard_alignment.evidence_level` | 标准门禁给出的证据等级，恒等于顶层 `level` |

### 规则证据层

1. 按 `。！？!?；;\n` 与句点切句，保留**精确字符偏移**；
2. 逐**子句**过滤否定/未来/背景语（`not / no / will / planned / 尚未 / 后续 …`，`no doubt`、`not only` 例外），
   这样 "Compared with X, efficiency improved 3%" 不会整句被丢；
3. 每个等级拆成 **core（定义对象）+ support（完成佐证）**：
   - 4 级 = 实验室/仿真环境 **且** 完成词（validated/results/synthesized/measured…）
   - 5 级 = 部件/分系统 **且** 模拟使用环境/集成
   - 6 级 = 原型/中试 **且** 测试/性能
   - 7 级 = 工程样机/现场试验/型式试验 **且** 通过/合格
   - 8 级 = 试运行/挂网/上线 **且** 全面测试
   - 9 级 = 在运/商业化/批量应用（要求有低阶段佐证）
   - 只有"仿真已开展但无结果"按协议上限记 3 级
4. 关键词统一走 `textmatch.term_pattern`：ASCII **词边界** + 复数，CJK 子串；
   因此 `test` 不会命中 `latest`，规则层与门禁层不会各用一套匹配语义。

### 标准门禁层

`config/qgdw_12566_2025.json` 给出九级定义和每级属性（技术状态 / 试验环境 / 活动 / 完成证据）：

```
Gate(l)      = 该级所有必需属性都有已完成证据，且无否定/未来/他人研究语境
TRL_evidence = max{ l : Gate(l) }        # 都不成立时为 null，不返回 0 级、不默认 1 级
TRL_final    = null                       # 标准 5.5 要求自评 + 专家评价
```

- 属性状态为 `met / partial / not_met`，附命中词与原文引文；
- 摘要通常不写明的属性标为 optional，并给出 `assumption` 供人工核对；
- 综述/政策/市场分析等不属于评价对象 → `not_applicable`（标准 1 范围，中英文标记均识别）；
- 模型等级高于证据 → `model_vs_evidence.verdict='model_over_grades'`，**不采信模型**；
- 下一级缺什么写在 `next_level_gap.missing_attributes`。

### 引文契约

每条引文都带 `start` / `end` / `truncated`：

```python
source = f"{title}. {abstract}"
assert source[q["start"]:q["end"]].startswith(q["quote"].rstrip("…"))
assert q["truncated"] is (len(source[q["start"]:q["end"]]) > len(q["quote"]))
```

超过 240 字的句子只截断显示，`start/end` 始终指向完整原句，可复原、可审计。

## 分类器后端（可选）

```bash
python -m energy_paper_trl --backend svm     --model-path /path/to/svm_local.joblib  --title '...' --abstract '...'
python -m energy_paper_trl --backend scibert --model-path /path/to/scibert_local     --title '...' --abstract '...'
python -m energy_paper_trl --backend scibert --model-path ... --no-explain          # 只出等级，跳过归因
```

- 9 类头（TRL 1–9）无法弃权；11 类头（+`not_applicable`/`insufficient_evidence`）可以弃权，
  弃权时 `level: null`，不会写成 10/11；
- 类别顺序优先读 `config.id2label`：语义名按名称映射，`LABEL_n` 按历史 9/11 类布局映射，
  宽度不认识时直接报错而不是猜；
- `max_length` 只接受显式参数或 `max_position_embeddings`（不会误取 transformers 的生成默认值 20）；
- 归因（`attribution`）分 SVM 的 TF-IDF 线性贡献与 SciBERT 的句子遮挡两种，整体标注
  `not_evidence: true`——它解释分类器，不构成论文证据；分数未校准（`score_calibrated: false`）。

## 输出示例

```json
{
  "level": 4,
  "level_source": "evidence",
  "levels": {"evidence": 4, "rule": 4, "model": null},
  "status": "public_evidence_preliminary",
  "evidence": [
    {"quote": "...", "start": 18, "end": 92, "truncated": false, "matched_terms": ["laboratory"], "kind": "core"}
  ],
  "standard_alignment": {
    "evidence_level": 4,
    "evaluation_status": "public_evidence_preliminary",
    "trl_model": null,
    "model_vs_evidence": {"gap": null, "verdict": "no_model_suggestion"},
    "next_level_gap": {"level": 5, "missing_attributes": ["technical_state"]},
    "gates_summary": [{"level": 1, "status": "not_met", "missing_attributes": ["technical_state"]}]
  },
  "uncertainty": {"band": "medium", "distress_score": 1, "reasons": ["仅使用标题和摘要"]}
}
```

默认返回 `gates_summary`（九级门禁摘要）以控制体积；需要逐属性审计时用
`analyze(..., include_gates=True)` 或 CLI `--full-gates`。

## 目录结构

```
src/energy_paper_trl/
  api.py          统一入口：唯一 level 契约、三层合并、不确定性
  rules.py        规则证据层：切句、子句过滤、core/support 分级
  standard.py     标准门禁层：属性 Gate、证据等级、对象检查
  textmatch.py    词边界匹配、引文区间、子句切分（三层共用）
  segmentation.py 精确句偏移
  explain.py      句子级归因与 rationale 模板
  model.py        SVM / SciBERT 适配器（惰性导入依赖）
  cli.py          命令行
config/qgdw_12566_2025.json   九级定义 + 默认门槛词表
tests/            规则、门禁、归因、接口契约测试
```

## 限制

- 只读标题与摘要：全文、测试记录、型式试验报告、专家审查不在接口内，`formal_certified_level` 恒为 `null`；
- 仅支持 editable 安装；标准层配置可用 `ENERGY_PAPER_TRL_STANDARD` 覆盖；
- 门槛关键词是本项目默认细则，不是标准统一规定，具体技术需专家按附录 C 配置；
- 输出是**公开证据初评**，不能作为正式 TRL 认证结论；
- 分类器分数未校准，且其训练语料与论文摘要存在分布差异，只作为第二意见。
