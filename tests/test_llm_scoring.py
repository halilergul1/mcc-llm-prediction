"""Label scoring on a tiny random model with a locally trained tokenizer (no downloads)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from mcc_llm import llm  # noqa: E402

LABELS = ["Clothing", "Gas stations", "Food and grocery", "Other"]
NEUTRAL = ["A", "B", "C", "D"]
SHARED = ["Clothing", "Gas stations", "Gas pumps", "Other"]          # two answers that start with the same token


def tiny():
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    corpus = ["USER: I made 9 purchases. Their categories: Food and grocery, Other, Clothing, Gas stations. ASSISTANT: "
              + lab + "." for lab in LABELS + NEUTRAL + SHARED] * 50
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=300, special_tokens=["<unk>", "<eos>"]))
    hf = transformers.PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<eos>", unk_token="<unk>")
    torch.manual_seed(0)
    lm = transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=len(hf), n_positions=256, n_embd=32,
                                                              n_layer=2, n_head=2)).eval()
    return hf, lm


def test_first_token_equals_first_term_of_sequence_score():
    tok, lm = tiny()
    prompts = ["USER: I made 9 purchases. Their categories: Food and grocery, Other. ASSISTANT:",
               "USER: I made 9 purchases. Their categories: Clothing, Clothing. ASSISTANT:"]
    ids = llm.answer_ids(tok, prompts[0], NEUTRAL)
    assert len({a[0] for a in ids}) == 4
    fast = llm.first_token_scores(tok, lm, prompts, NEUTRAL)
    exact = llm.sequence_scores(tok, lm, prompts, NEUTRAL)
    assert fast.shape == exact.shape == (2, 4)
    assert np.all(exact <= fast + 1e-5)                 # a longer string is never more likely than its first token
    # the exact score is the first-token score plus the log-probabilities of the remaining tokens
    p0 = tok(prompts[0]).input_ids
    rest = []
    for a in ids:
        seq = torch.tensor([p0 + a])
        logp = lm(input_ids=seq).logits[0].log_softmax(-1)
        rest.append(sum(logp[len(p0) + t - 1, a[t]].item() for t in range(1, len(a))))
    assert np.allclose(exact[0], fast[0] + np.array(rest), atol=1e-4)
    p = llm.to_probabilities(exact)
    assert np.allclose(p.sum(1), 1)
    report = llm.check_scorers(tok, lm, prompts, NEUTRAL)
    assert report["n"] == 2 and 0 <= report["argmax_agreement"] <= 1


def test_unclean_boundary_is_refused():
    tok, _ = tiny()
    with pytest.raises(ValueError, match="clean boundary"):
        llm.answer_ids(tok, "USER: Their categories: Other. ASSISTANT: ", LABELS)   # trailing space merges


def test_shared_first_token_is_refused():
    tok, lm = tiny()
    with pytest.raises(ValueError, match="share their first token"):
        llm.first_token_scores(tok, lm, ["USER: Their categories: Other. ASSISTANT:"], SHARED)
