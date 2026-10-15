<h1 align="center">AutoResearchBench</h1>

<p align="center">English · <a href="README_zh.md">简体中文</a></p>

<div align="center">

<strong>Reference code for inference and evaluation on the AutoResearchBench. (NIPS 2026)</strong>

<br />
<br />

<a href="#quick-start">
  <img src="https://img.shields.io/badge/Quick_Start-Run_the_Pipeline-111827?style=for-the-badge&logo=rocket&logoColor=white" alt="Quick Start" />
</a>
<a href="https://huggingface.co/datasets/Lk123/AutoResearchBench">
  <img src="https://img.shields.io/badge/Hugging_Face-Dataset-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face Dataset" />
</a>
<a href="#benchmark-data">
  <img src="https://img.shields.io/badge/Benchmark_Data-Download_%26_Decrypt-0F766E?style=for-the-badge&logo=databricks&logoColor=white" alt="Benchmark Data" />
</a>
<a href="#repository-map">
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

## Abstract

Autonomous scientific research is significantly advanced thanks to the development of AI agents. One key step in this process is finding the right scientific literature, whether to explore existing knowledge for a research problem, or to acquire evidence for verifying assumptions and supporting claims.

To assess AI agents' capability in driving this process, we present **AutoResearchBench**, a dedicated benchmark for autonomous scientific literature discovery.

AutoResearchBench consists of two complementary task types:

- **Deep Research**: requires tracking down a specific target paper through a progressive, multi-step probing process.
- **Wide Research**: requires comprehensively collecting a set of papers satisfying given conditions.

Compared to previous benchmarks on agentic web browsing, AutoResearchBench is distinguished along three dimensions: it is *research-oriented*, calling for in-depth comprehension of scientific concepts; *literature-focused*, demanding fine-grained utilization of detailed information; and *open-ended*, involving an unknown number of qualified papers and thus requiring deliberate reasoning and search throughout. These properties make AutoResearchBench uniquely suited for evaluating autonomous research capabilities, and extraordinarily challenging.

Even the most powerful LLMs, despite having largely conquered general agentic web-browsing benchmarks such as BrowseComp, achieve only 9.39% accuracy on Deep Research and 9.31% IoU on Wide Research, while many other strong baselines fall below 5%. We publicly release the dataset and evaluation pipeline to facilitate future research in this direction.


## Figures

Construction pipeline (high-level overview). Vector figure: [`assets/construction-pipeline.pdf`](assets/construction-pipeline.pdf).

![Construction pipeline overview](assets/construction-pipeline_preview.png)

Illustrative benchmark cases. Vector figure: [`assets/autoresearchbench-cases.pdf`](assets/autoresearchbench-cases.pdf).

![Benchmark case illustrations](assets/autoresearchbench-cases_preview.png)

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

1. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

2. Create an environment file:

```bash
cp example.env .env
```

3. Fill in the required fields in `.env`:

```bash
MODEL=your_model_name
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=your_api_base
INPUT_FILE=input_data/academic_deepsearch_example.jsonl
```

4. Run inference:

```bash
bash run_inference.sh
```

5. Run evaluation:

```bash
bash evaluate/run_evaluate.sh deep --input-file output_data/inference_output.jsonl
bash evaluate/run_evaluate.sh wide --input-file output_data/inference_output.jsonl --gt-file path/to/gt.jsonl
```

## Benchmark Data

The released benchmark bundle is hosted on the Hugging Face dataset repo [`Lk123/AutoResearchBench`](https://huggingface.co/datasets/Lk123/AutoResearchBench).

### 1. Download the released bundle

```bash
mkdir -p input_data

curl -L \
  -o input_data/AutoResearchBench.jsonl.obf.json \
  https://huggingface.co/datasets/Lk123/AutoResearchBench/resolve/main/AutoResearchBench.jsonl.obf.json
```

If you mirror the bundle into a private Hugging Face repo, add `-H "Authorization: Bearer ${HF_TOKEN}"` to the `curl` command.

### 2. Decrypt it locally

```bash
python3 decrypt_benchmark.py \
  --input-file input_data/AutoResearchBench.jsonl.obf.json \
  --output-file input_data/AutoResearchBench.jsonl
```

### 3. Point inference to the decrypted JSONL

```bash
INPUT_FILE=input_data/AutoResearchBench.jsonl
```

> [!NOTE]
> The released file on Hugging Face is the obfuscated bundle. Run inference on the decrypted `.jsonl`, not on the `.obf.json` file directly.

## Citation

If you use this benchmark or code, please cite the AutoResearchBench publication when available, and retain the dataset attribution required by the Hugging Face repository license.

## License

This repository is released under the Apache License 2.0. See [`LICENSE`](LICENSE) for details.

## Notes

- Inference automatically skips questions that already exist in the output JSONL file.
- `run_inference.sh` and `evaluate/run_evaluate.sh` both load configuration from `.env` by default. Set `AUTORESEARCHBENCH_ENV_FILE` to use a different environment file.
- Use `--verbose` on Python entrypoints when you need detailed debugging logs.
