# When2Tool: Interaction Round Budget Experiment

A cross-model subset reproduction of the CalculatorEnv Default baseline from *LLM Agents Already Know When to Call Tools — Even Without Reasoning*, with a maximum-round budget ablation.

- Paper: https://arxiv.org/abs/2605.09252
- Author repository: https://github.com/Trustworthy-ML-Lab/when2tool
- Author revision: `ef0cf1ea72eb8acdfd3fb1251c4b6a9a0131a23e`

## Run in Google Colab

Upload `When2Tool_DeepSeek_Round_Budget.ipynb` to Google Colab and execute its cells in order. The notebook checks out the author repository, installs dependencies, generates the data, writes the adapter and tests, and runs mock checks before any paid API requests. A CPU runtime is sufficient.

Enter your own DeepSeek API key at the hidden prompt. Never commit API keys. Live execution incurs API charges; mock tests do not.

1. Start with `STAGE='pilot'`: 3 tasks per difficulty, 9 distinct tasks, one run per condition.
2. For the formal experiment, set `STAGE='formal'` and retain `FORMAL_PER_DIFFICULTY=30`: 90 distinct tasks, three runs per condition.
3. Execute the result analysis and ZIP download cells.

Each execution creates a new timestamped output directory. Rerunning the experiment makes new paid requests; there is no automatic resume or retry. Request caps are enforced, but there is no hard monetary cap.

## Experimental scope

Both conditions use the same DeepSeek API model, Default prompting, `no_reasoning` policy, tools, and parser. The only changed parameter is the maximum interaction rounds: `baseline6` permits six rounds; `budget3` permits three. Six rounds is this project's baseline choice. Rejected model outputs also consume a round. No training or Probe&Prefill reproduction is performed.

Settings: `deepseek-flash`, temperature 1, maximum output 1024 tokens, internal thinking disabled, API endpoint `https://api.deepseek.com/chat/completions`. The model alias does not independently pin a model snapshot.

The API adaptation appends tool schema information to the prompt. The model and evaluation subset differ from the paper; results are not a strict numerical reproduction of the paper's full benchmark.

## Recorded results

Formal results cover 90 tasks (30 per difficulty), three runs per condition. Mean answer accuracy: baseline6 97.78% (sample SD 2.22 percentage points); budget3 91.48% (sample SD 0.64 percentage points). Total API requests: 553 versus 515. Reduced budget lowers accuracy as well as cost; it is not claimed to improve accuracy.

`results/formal/` contains episode logs, per-run metrics, same-prefix replay analysis, and the generated report. The same-prefix replay uses stored baseline outputs without additional API calls. Independent conditions involve generation randomness, so their entire difference cannot be attributed solely to budget.

## Files

- Notebook: complete Colab installation, tests, pilot, formal experiment, and analysis.
- `deepseek_rounds.py`: API adapter and round-budget runner (also embedded in the notebook).
- `test_rounds.py`: implementation tests (also embedded in the notebook).
- `results/`: recorded pilot and formal outputs.

Author code remains attributed to its original repository; consult that repository for its licensing terms. The notebook downloads the pinned author revision rather than redistributing the entire original codebase.
