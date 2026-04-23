<h1 align="center">AutoResearchBench</h1>

<p align="center">English · <a href="README_zh.md">简体中文</a></p>

<div align="center">

<strong>Reference code for inference and evaluation on the AutoResearchBench.</strong>

<br />
<br />

<a href="#quick-start">
  <img src="https://img.shields.io/badge/Quick_Start-Run_the_Pipeline-111827?style=for-the-badge&logo=rocket&logoColor=white" alt="Quick Start" />
</a>
<a href="https://huggingface.co/datasets/Lk123/Academic-DeepWide-Search">
  <img src="https://img.shields.io/badge/Hugging_Face-Dataset-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face Dataset" />
</a>
<a href="#benchmark-data">
  <img src="https://img.shields.io/badge/Benchmark_Data-Download_%26_Decrypt-0F766E?style=for-the-badge&logo=databricks&logoColor=white" alt="Benchmark Data" />

<br />
<br />

<img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+" />
<img src="https://img.shields.io/badge/Inference-Batch-0F766E?style=flat-square" alt="Batch Inference" />
<img src="https://img.shields.io/badge/Evaluation-Deep_%2B_Wide-1D4ED8?style=flat-square" alt="Deep and Wide Evaluation" />
<img src="https://img.shields.io/badge/Search-DeepXiv_%2B_Web-7C3AED?style=flat-square" alt="Search Backends" />
<img src="https://img.shields.io/badge/Release-Obfuscated_Bundle-9A3412?style=flat-square" alt="Obfuscated Bundle" />

</div>

## Abstract

Autonomous scientific research is significantly advanced thanks to the development of AI agents. One key step in this process is finding the right scientific literature, whether to explore existing knowledge for a research problem, or to acquire evidence for verifying assumptions and supporting claims. 
To assess AI agents' capability in driving this process, we present **AutoResearchBench**, a dedicated benchmark for autonomous scientific literature discovery.

Two complementary tracks are evaluated:

- **Deep Research** (*precision-oriented retrieval*): locate the unique paper that satisfies a set of obfuscated, detail-heavy constraints (including evidence that may appear deep in the main text, appendix, or tables).
- **Wide Research** (*recall-oriented set construction*): given a topic and constraints, collect **all** papers that meet the specification—analogous to exhaustive coverage in a structured literature survey.

Empirical reports in the accompanying materials indicate that contemporary frontier models remain **far below saturation** on these tracks (e.g., on the order of **single-digit accuracy** for Deep Search and **single-digit IoU** for Wide Search), underscoring that strong performance on generic web browsing does not automatically transfer to these settings.


## Figures

Construction pipeline (high-level overview). Vector figure: [`assets/construction-pipeline.pdf`](assets/construction-pipeline.pdf).

![Construction pipeline overview](assets/construction-pipeline_preview.png)

Illustrative benchmark cases. Vector figure: [`assets/Academic-Search-Agent-Benchmark-cases.pdf`](assets/Academic-Search-Agent-Benchmark-cases.pdf).

![Benchmark case illustrations](assets/Academic-Search-Agent-Benchmark-cases_preview.png)

Main experimental results reported with the DeepXiv search tool (end-to-end systems evaluated separately in the table’s protocol). Raster export of the paper’s summary table:

![Main experimental results](assets/main_results_table.png)

## Repository Map

| Icon | Component | Purpose |
| --- | --- | --- |
| 🚀 | `run_inference.sh` + `inference.py` | Main batch inference entrypoint with `.env`-driven configuration. |
| 🔎 | `tool_deepxivsearch.py` + `tool_websearch.py` | Search backends for academic retrieval and general web retrieval. |
| 🧠 | `prompts.py` + `utils.py` | Shared prompting logic, model client wiring, and JSONL helpers. |
| 📊 | `evaluate/evaluate_deep_search.py` + `evaluate/evaluate_wide_search.py` | Deep-search judging and wide-search retrieval metrics. |
| 🔓 | `decrypt_benchmark.py` + `benchmark_crypto.py` | Local bundle restoration from the released `.obf.json` file back to plaintext JSONL. |

## Quick Start

1. Create an environment file:

```bash
cp example.env .env
```

2. Fill in the required fields in `.env`:

```bash
MODEL=your_model_name
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=your_api_base
INPUT_FILE=input_data/academic_deepsearch_example.jsonl
```

3. Run inference:

```bash
bash run_inference.sh
```

4. Run evaluation:

```bash
bash evaluate/run_evaluate.sh deep --input-file output_data/inference_output.jsonl
bash evaluate/run_evaluate.sh wide --input-file output_data/inference_output.jsonl --gt-file path/to/gt.jsonl
```

## Benchmark Data

The released benchmark bundle is hosted on the Hugging Face dataset repo [`Lk123/Academic-DeepWide-Search`](https://huggingface.co/datasets/Lk123/Academic-DeepWide-Search).

### 1. Download the released bundle

```bash
mkdir -p input_data

export HF_TOKEN=your_hf_token  # required if the dataset repo is private
curl -L \
  -H "Authorization: Bearer ${HF_TOKEN}" \
  -o input_data/Academic-DeepWide-Search.jsonl.obf.json \
  https://huggingface.co/datasets/Lk123/Academic-DeepWide-Search/resolve/main/Academic-DeepWide-Search.jsonl.obf.json
```

### 2. Decrypt it locally

```bash
python3 decrypt_benchmark.py \
  --input-file input_data/Academic-DeepWide-Search.jsonl.obf.json \
  --output-file input_data/Academic-DeepWide-Search.jsonl
```

### 3. Point inference to the decrypted JSONL

```bash
INPUT_FILE=input_data/Academic-DeepWide-Search.jsonl
```

> [!NOTE]
> The released file on Hugging Face is the obfuscated bundle. Run inference on the decrypted `.jsonl`, not on the `.obf.json` file directly.

## Citation

If you use this benchmark or code, please cite the Academic DeepWide Search publication when available, and retain the dataset attribution required by the Hugging Face repository license.

## Notes

- Inference automatically skips questions that already exist in the output JSONL file.
- `run_inference.sh` and `evaluate/run_evaluate.sh` both load configuration from `.env` by default.
- Use `--verbose` on Python entrypoints when you need detailed debugging logs.
