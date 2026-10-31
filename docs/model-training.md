# 当前版本模型的训练说明

> 适用版本：核心版 `energy-paper-trl` 0.4.x（`src/energy_paper_trl/model.py`）。
> 本文只说明**当前版本**的两个可选分类器（SVM / SciBERT）是怎么训练的；训练数据、训练脚本与权重
> **不随本仓库交付**，此处给出可核对的口径与指标。

## 1. 两个分类器

| 后端 | 形式 | 输入 | 输出 |
| --- | --- | --- | --- |
| `--backend svm` | TF-IDF + LinearSVC | 标题 + 摘要 | 11 类标签 + decision margin |
| `--backend scibert` | SciBERT 编码器 + 分类头 | 标题 + 摘要 | 11 类标签 + softmax 分数 |

两者是**并列的第二意见**：最终采信等级由规则证据层与标准门禁层决定（`level_source='evidence'`）；
模型等级高于证据时不采信，见 README「三层结构」。

## 2. 训练数据

- **来源**：OpenAlex 论文的 `title + "\n" + abstract`（英文为主），从历史论文语料中按目标检索取样；
  取样只决定阅读顺序，不改变标签；已标注记录与旧诊断集全部排除。
- **标注**：助手语义标注（`label_source = assistant_semantic_annotation`），**不是独立专家金标**；
  同一助手完成边界与一致性复核。标注要求只读标题与摘要、给出精确引文偏移、对象边界、缺失证据与
  语义项目族。
- **规模**（2026-10-23 训练批次）：

  | 环节 | 数量 |
  | --- | ---: |
  | 累计标注 | 833（400 旧 + 433 新） |
  | 候选训练池 | 633 |
  | 可用池 | 601 |
  | 训练 / 验证 | **275 / 55** |
  | 留作备选（reserve） | 271 |
  | 排除（无可用训练价值） | 32 |
  | 既有诊断集（不参与训练） | 50 + 50 + 100 |

- **类别配额**：11 类 × 25 篇训练、11 类 × 5 篇验证。协议是 **evidence-first**：不合成论文、
  不复制行、不按关键词自动贴标、不为凑配额而升级标签；多数类多余标注留在 reserve 池。
- **划分**：标注锁定后按标签分层、固定 seed 42；**组级隔离**（doc id、精确标题/摘要、标题相似度
  ≥ 0.86、摘要 Jaccard ≥ 0.7、手工项目族归入同一折），训练/验证组无重叠（270 / 55 组），
  与既有诊断集也无重叠。
- **文本预处理**：`mask(title + "\n" + abstract)` —— 把显式 TRL 数字短语（如 “TRL 4”、
  “technology readiness level 5”）替换为 `readiness stage`，避免标签从原文泄漏进特征；
  推理时使用同一清洗函数。

## 3. 标签空间（11 类）

| 索引（SciBERT `id2label`） | 标签 | 含义 |
| ---: | --- | --- |
| 0–8 | `1` … `9` | Q/GDW 12566—2025 表 1 的 TRL 1–9 |
| 9 | `not_applicable` | 不属于评价对象（综述、政策、市场分析等） |
| 10 | `insufficient_evidence` | 证据不足，无法定级 |

标签由标注的 `reference_level`（1–9）与 `reference_status`（两个弃权状态）合成：
`label = reference_level or reference_status`。两个弃权类不会写成 10/11。

## 4. 模型 A：SVM

```
Pipeline([
  ("vect", TfidfVectorizer(analyzer="word", ngram_range=(1,3), max_features=50000,
                            lowercase=True, norm="l2", sublinear_tf=False)),
  ("clf",  LinearSVC(C=10.0, class_weight=None, loss="squared_hinge",
                     penalty="l2", dual="auto", max_iter=10000, tol=1e-4)),
])
```

- **选参**：`C ∈ {0.1, 1, 10}` × `class_weight ∈ {None, balanced}`，按验证集 macro-F1 选择，
  并列时看 decision accuracy；最终 `C=10`、`class_weight=None`。
- **验证集表现**：macro-F1 **0.3224**，decision accuracy **0.3636**（55 篇、11 类各 5 篇）。
- 类别顺序为 sklearn 排序（`1..9, insufficient_evidence, not_applicable`）；适配器按**标签名**
  映射，不依赖顺序。

## 5. 模型 B：SciBERT

- **基座**：`allenai/scibert_scivocab_uncased`（revision `24f92d3`），分类头 `num_labels=11`，
  `id2label` 按上表。
- **微调**：`max_length=512`；候选 seed 42 / 43；类别权重
  `min(10, sqrt(N / (11 × n_c)))`（本批均衡后约为 1）。
- **选模**：按验证 macro-F1 → decision agreement → 更低的未加权交叉熵；**从不使用测试集选模**
  （`test_used_in_selection = false`）。最终取 seed 43、第 7 个 epoch。
- **验证集表现**：macro-F1 **0.4693**，decision accuracy **0.5091**（同一 55 篇）。

## 6. 已知限制（与输出口径相关）

- 标签是**助手语义标注**，不是专家金标；验证集每类仅 5 篇，指标的置信区间很宽。
- 训练集多数类为 `1`，模型在分布外文本上可能偏向低等级；两个弃权类只在这 55 篇验证上评估过。
- 分数**未校准**（`score_calibrated: false`），只能作相对排序，不能当概率用。
- 只读标题 + 摘要；显式 TRL 短语被掩码，模型无法利用论文自述的等级。
- 因此模型等级只作第二意见：最终 `level` 来自证据层，模型高于证据时不采信。

## 7. 附：可核对的训练记录

以下文件属于训练批次产物（**不随本仓库交付**），字段与本文一一对应：

| 文件 | 内容 |
| --- | --- |
| `PROTOCOL.json` | 标注与建模协议、目标配额、split 策略 |
| `DATASET_SUMMARY.json` | 标注/可用池/训练/验证/reserve/excluded 数量与逐类计数 |
| `SPLIT_AUDIT.json` | 组级划分审计（seed、阈值、组数、重叠检查） |
| `SVM_TRAINING.json` | SVM 网格搜索与选中参数、验证预测 |
| `SCIBERT_TRAINING.json` | SciBERT 训练历史、选模结果、验证预测 |
| `MODEL_CARD.json` | 类别空间、`max_length`、选中超参、训练/验证规模、`test_used_in_selection` |
