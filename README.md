# Reinforcement Learning with Human Feedback (RLHF)

Coursework notebooks for Advanced Topics in AI & ML that walk through the three stages of an RLHF pipeline on the [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) instruction dataset:

1. **SFT (supervised fine-tuning):** train a causal language model on instruction and answer text.
2. **RM (reward model):** add a linear head to a transformer so it outputs a single score for a text.
3. **PPO (proximal policy optimization):** update a policy model using scores from a reward model.

The same pipeline is set up for GPT-2, GPT-2 with LoRA, GPT-Neo 1.3B and Falcon-7B Instruct.

> **Before you read the results:** the pipeline is a learning implementation, and most saved outputs come from GPT-2, whichever model the notebook names. See [Limitations](#limitations).

## Repository layout

Each folder holds one notebook per stage. Open them in the order SFT, RM, PPO.

| Folder | `model_name` in the notebooks | SFT | RM | PPO |
| --- | --- | --- | --- | --- |
| [`gpt2/`](gpt2/) | `gpt2` | [SFT](gpt2/SFT_gpt2_alphaca-1.ipynb) | [RM](gpt2/RM_gpt2_alphaca-1.ipynb) | [PPO](gpt2/PPO_gpt2_alphaca-1.ipynb) |
| [`gpt2 with lora/`](gpt2%20with%20lora/) | `gpt2` | [SFT](gpt2%20with%20lora/SFT_gpt2_alphaca_lora.ipynb) | [RM](gpt2%20with%20lora/RM_gpt2_alphaca_lora.ipynb) | [PPO](gpt2%20with%20lora/PPO_gpt2_alphaca-1.ipynb) |
| [`EleutherAI-gpt-neo-1.3B/`](EleutherAI-gpt-neo-1.3B/) | `EleutherAI/gpt-neo-1.3B` | [SFT](EleutherAI-gpt-neo-1.3B/SFT_gpt_neo_1_3B_alphaca-1.ipynb) | [RM](EleutherAI-gpt-neo-1.3B/RM_gpt_neo_1_3B_alphaca-1-3.ipynb) | [PPO](EleutherAI-gpt-neo-1.3B/PPO_gpt_neo_1_3B_alphaca-1.ipynb) |
| [`tiiuae-falcon-7b/`](tiiuae-falcon-7b/) | `tiiuae/falcon-7b-instruct` | [SFT](tiiuae-falcon-7b/SFT_falcon_7b_instruct_alphaca-1.ipynb) | [RM](tiiuae-falcon-7b/RM_falcon_7b_instruct_alphaca-1.ipynb) | [PPO](tiiuae-falcon-7b/PPO_falcon_7b_instruct_alphaca-1.ipynb) |

`gpt2 with lora/` also contains result images:

- `0.001/`: the loss curve and sample generations from the LoRA SFT run, which used a learning rate of `1e-3`.
- `images/`: screenshots of a 3-epoch reward model run. Their numbers don't match the output saved in any RM notebook, so the run they came from isn't identified.

## How each stage works

### SFT

Examples are formatted as `Instruction: ...`, an optional `Input: ...`, and `Answer:`. They are tokenized and trained with AdamW, and the notebook then plots the loss, saves the model and generates text from sample prompts.

### LoRA

Only the SFT notebook in `gpt2 with lora/` uses LoRA, with this configuration:

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["c_attn"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

### RM

A transformer's last-token hidden state goes through a linear layer to produce one score. Training uses AdamW and MSE loss against a target of `1` for every example. The notebooks also report reward statistics and plot their distribution.

### PPO

A policy samples 5 tokens per prompt, and the reward model scores the result. The advantage is the difference between the scores for the policy's sample and a sample from an older copy of the policy. The loss is the clipped PPO surrogate.

## Settings

These values come from the code currently in the notebooks. Saved outputs may come from different settings, as noted under [Limitations](#limitations).

| Stage | Setting | GPT-2 | GPT-2 + LoRA | GPT-Neo 1.3B | Falcon-7B |
| --- | --- | --- | --- | --- | --- |
| SFT | Examples used | 5,000 | 10,000 | 10,000 | All |
| SFT | Learning rate | `5e-5` | `1e-3` | `2e-5` | `5e-5` |
| SFT | Epochs | 1 | 3 | 3 | 1 |
| RM | Examples used | All | All | 10,000 | All |
| RM | Learning rate | `1e-5` | `1e-5` | `2e-5` | `1e-5` |
| RM | Epochs | 1 | 1 | 3 | 1 |
| PPO | Examples used | All | All | 10,000 | All |

Other settings are shared:

- **Validation split:** 20% in every notebook.
- **SFT and RM batch size:** 8.
- **PPO:** batch size 4, 3 epochs, learning rate `1e-5`, clip range 0.2, maximum prompt length 32 tokens.

## Saved results

The results are grouped by the notebook that produced them. Read the GPT-Neo row as a GPT-2 run with different settings, for the reasons given under [Limitations](#limitations).

### SFT

| Notebook | Epochs | Train loss | Validation loss |
| --- | --- | --- | --- |
| `gpt2/` | 1 | 2.0727 | 1.9608 |
| `gpt2 with lora/` | 3 | 2.2289 → 2.1394 | 2.1519 → 2.1383 |
| `EleutherAI-gpt-neo-1.3B/` | 3 | 2.3870 → 1.8985 | 2.1074 → 2.0639 |

The `tiiuae-falcon-7b/` output is identical to `gpt2/`, so it isn't listed separately.

![Training and validation loss for the GPT-2 LoRA SFT run over 3 epochs](gpt2%20with%20lora/0.001/8.png)

Generated answers often repeat themselves. The LoRA model, for example, answered a summarization prompt by repeating the input sentence several times.

### RM

| Notebook | Epochs | Train loss | Validation loss |
| --- | --- | --- | --- |
| `gpt2/` | 1 | 0.0237 | 0.00035 |
| `EleutherAI-gpt-neo-1.3B/` | 3 | 0.1892 → 0.0046 | 0.0043 → 0.0014 |

The `gpt2 with lora/` and `tiiuae-falcon-7b/` output is identical to `gpt2/`. Low loss here doesn't show that the model can judge quality, because every target is `1`.

### PPO

No PPO notebook has a completed run saved:

- The GPT-Neo notebook shows epoch 1 of 3 finishing with a loss of 0.4671, and its output then stops.
- The other three notebooks show only 22 logged loss values, with no finished epoch.
- None has saved output from its plotting or evaluation cells.

## Limitations

**Most saved outputs come from GPT-2**
- Every PPO notebook prints an architecture with 12 `GPT2Block` layers and a 768-dimension reward head, including the GPT-Neo and Falcon notebooks.
- The GPT-Neo SFT notebook prints a `GPT2LMHeadModel` and a save path of `sft_model-gpt2-alpaca`.
- The Falcon SFT and RM outputs are identical to GPT-2's, and the Falcon SFT also saves to `sft_model-gpt2-alpaca`.
- The GPT-Neo RM notebook prints no architecture, so its model can't be confirmed.
- There is no valid comparison between the four models yet. Getting one means rerunning the GPT-Neo and Falcon notebooks.

**The pipeline doesn't use human feedback yet**
- The reward model trains on a constant target of `1`, with no preference data or human ratings, so it can't learn to tell good answers from bad ones.

**The stages aren't connected**
- PPO builds a new reward model instead of loading the one trained in RM, so its scores come from an untrained head.
- PPO starts its policy from the pretrained base model instead of the SFT checkpoint.
- The older policy copy is never updated during training.

**Other differences between code and output**
- The saved output of `gpt2/SFT` shows 49,401 training and 2,601 validation examples, but the code selects 5,000 examples.
- The saved output of the `gpt2/` RM notebook shows the same 49,401 and 2,601 split, but the code uses a 20% validation split.
- The RM and PPO notebooks in `gpt2 with lora/` have the same code as those in `gpt2/` and don't use LoRA, despite the `_lora` file name on the RM notebook.

## Running the notebooks

The notebooks need Python 3.12 or later, because they use f-strings with nested quotes of the same type, and a GPU is strongly recommended.

```bash
pip install torch transformers datasets peft accelerate matplotlib numpy jupyter
```

The notebooks contain commented-out install lines, including `transformers==4.37.1` and a CUDA 12.8 build of PyTorch. The versions used for the saved runs aren't otherwise recorded, so you may need to adjust them.

The SFT notebooks save to a path at the filesystem root (`/sft_model-...`), so change that path if you don't have write access there. Checkpoints aren't included in this repository.
