<h1 align="center">Academic DeepWide Search</h1>

<p align="center"><a href="README.md">English</a> · 简体中文</p>

<div align="center">

<strong>Academic DeepWide Search 基准的推理与评测参考实现。</strong>

<br />
<br />

<a href="#快速开始">
  <img src="https://img.shields.io/badge/Quick_Start-Run_the_Pipeline-111827?style=for-the-badge&logo=rocket&logoColor=white" alt="Quick Start" />
</a>
<a href="https://huggingface.co/datasets/Lk123/Academic-DeepWide-Search">
  <img src="https://img.shields.io/badge/Hugging_Face-Dataset-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face Dataset" />
</a>
<a href="#基准数据">
  <img src="https://img.shields.io/badge/Benchmark_Data-Download_%26_Decrypt-0F766E?style=for-the-badge&logo=databricks&logoColor=white" alt="Benchmark Data" />
</a>
<a href="#仓库结构">
  <img src="https://img.shields.io/badge/Repository_Map-Code_Guide-2563EB?style=for-the-badge&logo=github&logoColor=white" alt="Repository Map" />
</a>

<br />
<br />

<img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+" />
<img src="https://img.shields.io/badge/Inference-Batch-0F766E?style=flat-square" alt="Batch Inference" />
<img src="https://img.shields.io/badge/Evaluation-Deep_%2B_Wide-1D4ED8?style=flat-square" alt="Deep and Wide Evaluation" />
<img src="https://img.shields.io/badge/Search-DeepXiv_%2B_Web-7C3AED?style=flat-square" alt="Search Backends" />
<img src="https://img.shields.io/badge/Release-Obfuscated_Bundle-9A3412?style=flat-square" alt="Obfuscated Bundle" />

</div>

## 概要

Academic DeepWide Search 面向**科学文献检索**场景：可信答案往往依赖**全文层面的证据**、**细粒度对齐**以及沿**引用关系的多跳推理**，而非仅依赖标题、摘要或浅层关键词匹配。

基准包含两类互补任务：

- **Deep Search（深度定向检索）**：在给定一组经过模糊化、细节密集的约束下，定位**唯一**满足条件的论文；证据可能分布在正文深处、附录或图表中。
- **Wide Search（广度集合构造）**：在给定主题与约束时，尽可能**完整**地枚举所有符合条件的论文，更接近结构化综述工作的“覆盖面”要求。

配套材料中的实证结果表明，当前主流前沿模型在该任务族上距离“充分可用”仍有明显差距（例如：在报告协议下，Deep Search 的准确率与 Wide Search 的 IoU 大致处于**个位数百分比**量级）。这在一定程度上说明：**通用网页浏览或开放域检索上的优势，并不能自动迁移到上述科研文献检索设定**。

## 图示

构造流程（示意）。矢量版：[`assets/construction-pipeline.pdf`](assets/construction-pipeline.pdf)。

![构造流程示意](assets/construction-pipeline_preview.png)

Benchmark 案例示意。矢量版：[`assets/Academic-Search-Agent-Benchmark-cases.pdf`](assets/Academic-Search-Agent-Benchmark-cases.pdf)。

![Benchmark 案例示意](assets/Academic-Search-Agent-Benchmark-cases_preview.png)

主实验结果汇总（表中在统一协议下以 DeepXiv 检索工具评测；端到端系统另行列出）。下图来自论文材料的表格导出：

![主实验结果](assets/main_results_table.png)

## 仓库结构

| 图标 | 组件 | 作用 |
| --- | --- | --- |
| 🚀 | `run_inference.sh` + `inference.py` | 批推理入口，配置由 `.env` 驱动。 |
| 🔎 | `tool_deepxivsearch.py` + `tool_websearch.py` | 学术检索与通用网页检索后端。 |
| 🧠 | `prompts.py` + `utils.py` | 共享提示词、模型客户端与 JSONL 工具函数。 |
| 📊 | `evaluate/evaluate_deep_search.py` + `evaluate/evaluate_wide_search.py` | Deep Search 判定与 Wide Search 检索指标。 |
| 🔓 | `decrypt_benchmark.py` + `benchmark_crypto.py` | 将发布的 `.obf.json` 在本地还原为明文 JSONL。 |
| 🗂️ | `input_data/` + `output_data/` | 示例输入与推理输出。 |

## 快速开始

1. 复制环境变量模板：

```bash
cp example.env .env
```

2. 在 `.env` 中填写必要字段：

```bash
MODEL=your_model_name
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=your_api_base
INPUT_FILE=input_data/academic_deepsearch_example.jsonl
```

3. 运行推理：

```bash
bash run_inference.sh
```

4. 运行评测：

```bash
bash evaluate/run_evaluate.sh deep --input-file output_data/inference_output.jsonl
bash evaluate/run_evaluate.sh wide --input-file output_data/inference_output.jsonl --gt-file path/to/gt.jsonl
```

## 基准数据

公开发布的数据包托管于 Hugging Face 数据集仓库 [`Lk123/Academic-DeepWide-Search`](https://huggingface.co/datasets/Lk123/Academic-DeepWide-Search)。

### 1. 下载发布包

```bash
mkdir -p input_data

export HF_TOKEN=your_hf_token  # 若仓库为私有可能需要
curl -L \
  -H "Authorization: Bearer ${HF_TOKEN}" \
  -o input_data/Academic-Deep-Search-CR-0331.jsonl.obf.json \
  https://huggingface.co/datasets/Lk123/Academic-DeepWide-Search/resolve/main/Academic-Deep-Search-CR-0331.jsonl.obf.json
```

### 2. 本地解密

```bash
python3 decrypt_benchmark.py \
  --input-file input_data/Academic-Deep-Search-CR-0331.jsonl.obf.json \
  --output-file input_data/Academic-Deep-Search-CR-0331.jsonl
```

### 3. 将推理输入指向解密后的 JSONL

```bash
INPUT_FILE=input_data/Academic-Deep-Search-CR-0331.jsonl
```

> [!NOTE]
> Hugging Face 上发布的是混淆包；请对已解密的 `.jsonl` 运行推理，不要直接对已混淆的 `.obf.json` 调用推理流程。

## 引用

若在研究中使用本基准或本仓库代码，请在正式论文发表后按其版本信息引用；并遵守 Hugging Face 数据集页面的许可与署名要求。

## 说明

- 推理会跳过输出 JSONL 中已存在的题目。
- `run_inference.sh` 与 `evaluate/run_evaluate.sh` 默认从 `.env` 读取配置。
- 调试时可在 Python 入口使用 `--verbose` 获取更详细的日志。
