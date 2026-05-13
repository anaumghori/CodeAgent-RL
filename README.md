# CodeAgent-RL

This project is a replication of the reinforcement learning pipeline described in the Composer 2 report, focused on tool-using code agents. The system runs a full end-to-end loop with vLLM for rollout generation, DeepSpeed ZeRO-3 for training and live weight syncing between training and inference on a 3 GPU setup built around H200s.

The goal here was not to build a polished framework first. It was to understand the actual systems work behind this kind of training setup and make the whole thing run in one real pipeline.

More detailed writeup is available in [the blog post](https://anaumghori.github.io/blog_template.html?slug=CodeAgent-RL).

- Model: [NousResearch/Hermes-4-14B](https://huggingface.co/NousResearch/Hermes-4-14B)
- Datasets: [SWE-rebench-V2](https://huggingface.co/datasets/nebius/SWE-rebench-V2) and [CodeContests-O](https://huggingface.co/datasets/caijanfeng/CodeContests-O)

## How to run

First install Modal:

```bash
uv add modal
```

or

```bash
pip install modal
```

Then authenticate your Modal account:

```bash
modal token set --token-id <your-token-id> --token-secret <your-token-secret>
```

Then copy `.env.example` to `.env` and fill in the required API keys and tokens there before running the pipeline.

After that, run the training pipeline with:

```bash
modal run src/modal_app.py
```

If you want to resume from a saved checkpoint:

```bash
modal run src/modal_app.py --resume-from checkpoint-14
```

## Notes

- The pipeline is designed to run inside Modal.
- You should also have the usual project credentials available through `.env` if you want Hugging Face access, checkpoint sync, and W&B logging to work properly.
- The codebase currently assumes a 3-GPU layout with 2 training GPUs and 1 inference GPU.
