# HV-MMBench: Benchmarking MLLMs for Human-Centric Video Understanding

<p align="center">
    🤓 <a href="https://github.com/Fantasyele/HV-MMBench">GitHub</a> &nbsp&nbsp | 📑 <a href="https://arxiv.org/">Paper</a> &nbsp&nbsp | 🤗 <a href="https://huggingface.co/datasets/ccaiyuxuan/HVMMBench/tree/main">Hugging Face (HV-MMBench))</a>&nbsp&nbsp
<br>

-----

## Quickstart

1. Refer to [QwenLM/Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) for environment preparation.
``` sh
pip install transformers==4.51.3 accelerate
pip install qwen-vl-utils[decord]
pip install -U flash-attn --no-build-isolation
```
2. Download pre-trained [Qwen2.5-VL models](https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5) for inference.
3. Quick Inference
``` sh
python eval/human/inference_video_humanbench.py \
    --model-path $MODEL_PATH \
    --question-file $QAFile_DIR/$QA_Type/cleaned_qa_pair.json \
    --video-folder $Video_DIR \
    --answer-file $EVAL_DIR/$QA_Type/results.json
```
4. Evaluation
``` sh
python eval/human/eval_video_humanbench.py \
    --pred-path $EVAL_DIR/$QA_Type/results.json \
    --fixed-path $EVAL_DIR/$QA_Type/results_fixed.json \
    --save_csv_path $EVAL_DIR/$QA_Type/all_results.json
```