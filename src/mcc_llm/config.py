"""Settings of the paper, in one place.

The values follow Sections 3.2 to 3.8 and Table 2 of the paper. Scripts use them
as defaults; most can be changed on the command line.
"""

# --------------------------------------------------------------------------- labels
CATEGORIES = ("Clothing", "Gas stations", "Food and grocery", "Other")  # class codes 0, 1, 2, 3
NUM_CLASSES = len(CATEGORIES)

# Four-digit merchant category codes (MCC, ISO 18245) of the three named classes; every other code is Other.
# Gas stations: service stations (5541), automated fuel dispensers (5542) and fuel dealers (5983).
# Clothing: apparel and accessory stores (5600-5699), wholesale apparel (5137, 5139) and sporting goods (5941).
# Food and grocery: food stores (5400-5499). Eating places (5811-5814) are Other.
# The class names are the words the prompts use.
MCC_CATEGORIES = {
    "Gas stations": (5541, 5542, 5983),
    "Clothing": tuple(range(5600, 5700)) + (5137, 5139, 5941),
    "Food and grocery": tuple(range(5400, 5500)),
}

# --------------------------------------------------------------------------- cohorts and splits (Sections 3.1 and 3.2)
MIN_TRANSACTIONS = 10          # transactions per customer in the observation period
DIVERSITY_WINDOW = 10          # the last 10 transactions: 9 inputs and the target
MIN_DISTINCT_CATEGORIES = 2    # at least 2 distinct categories ...
DIVERSITY_ON_INPUTS = True     # ... among the 9 inputs, so the filter never looks at the target.
                               # False counts the target too (the earlier rule).
N_TRAIN_POOL = 50_000          # Bank A customers drawn from the eligible cohort
VAL_FRACTION = 0.2             # 40,000 training and 10,000 validation customers
N_BANK_A_TEST = 10_000         # in-bank test customers, never in training or validation
N_TEST = 1_000                 # prepare_data.py --earlier-rule only: size of the single test draw
SEED = 42

# Evaluation splits. The non-LLM models are scored on all of them when they are trained. The LLMs are scored
# on the first group while they are trained and on the second group once, after the freeze of ANALYSIS_PLAN.md.
PRE_SPLITS = ("val", "bank_a_test", "bank_b_dev")
CONFIRM_SPLITS = ("bank_b_confirm", "bank_b_unfiltered")
EVAL_SPLITS = PRE_SPLITS + CONFIRM_SPLITS

# --------------------------------------------------------------------------- protocol (Section 3.8)
TRAIN_LENGTH = 9               # 9 input transactions; the 10th, most recent one is the target
EVAL_LENGTHS = (4, 7, 9, 14)   # a customer enters length X only with at least X + 1 transactions
DEMOGRAPHIC_FIELDS = ("gender", "age", "marital_status", "education", "occupation", "income_group")

# --------------------------------------------------------------------------- amounts in the prompt
CURRENCY = "TL"                # printed as "TL 300.00"; the amounts are nominal Turkish lira
DEFLATE_TO = None              # nominal amounts. A month such as "2013-04" (with prepare_data.py --cpi) deflates them.

# --------------------------------------------------------------------------- sequential baselines (Section 3.6)
# LSTM and CNN read learned embeddings of the class codes
LSTM = {"hidden_size": 128, "embedding_dim": 32}
CNN = {"embedding_dim": 32, "channels": 64}  # three causal dilated convolutions (1, 2, 4), last position
SASREC = {"hidden_units": 50, "num_blocks": 2, "num_heads": 1, "dropout_rate": 0.2, "maxlen": 9}
DROS = {"hidden_size": 64, "state_size": 9, "dropout": 0.1}

NN_TRAINING = {"batch_size": 64, "lr": 1e-3, "max_epochs": 30, "patience": 5}        # LSTM and CNN
SASREC_TRAINING = {"batch_size": 128, "lr": 1e-3, "max_epochs": 30, "patience": 5}
DROS_TRAINING = {"batch_size": 256, "lr": 1e-3, "weight_decay": 1e-6, "max_epochs": 50, "patience": 5}

# Grids searched by the training scripts on the validation split. Each grid contains the configuration above.
# Loss weighting is a grid option, and the configuration is selected by the primary metric of ANALYSIS_PLAN.md.
CLASS_WEIGHTS = ("none", "inverse")
SELECTION_METRIC = "Macro-F1"  # or "F1 (w)"
BASELINE_SEEDS = (42, 2, 3, 4, 5)   # the grid is searched with the first seed; the selected configuration is refitted with each
LSTM_GRID = {"hidden_size": (64, 128, 256), "lr": (1e-3, 5e-4)}
CNN_GRID = {"lr": (1e-3, 5e-4), "batch_size": (64, 128)}
SASREC_GRID = {"hidden_units": (50, 64), "dropout_rate": (0.2, 0.5), "lr": (1e-3, 5e-4)}
DROS_GRID = {"alpha": (0.01, 0.1, 0.5, 1.0), "beta": (0.5, 1.0, 5.0)}
DROS_REPO = "https://github.com/YangZhengyi98/DROS"
DROS_COMMIT = "44ff99d"

# --------------------------------------------------------------------------- LLM fine-tuning (Section 3.4, Table 2)
BASE_MODELS = {
    "qwen": "Qwen/Qwen3.5-9B",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}
LORA = {
    "r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}
SFT = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,   # effective batch size 16
    "num_train_epochs": 3,              # fixed; the final checkpoint is used, no early stopping
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "optim": "paged_adamw_8bit",
    "max_grad_norm": 0.3,
    "bf16": True,
    "max_length": 512,
}
CHECKPOINT_STEPS = 500         # a checkpoint every 500 steps, so training can resume
MAX_NEW_TOKENS = 10            # greedy free generation, used as a check; predictions come from label scoring
FEW_SHOT_PER_CLASS = 4         # 16-shot prompts for the raw models

# --------------------------------------------------------------------------- evaluation (Section 3.8)
N_BOOTSTRAP = 10_000
