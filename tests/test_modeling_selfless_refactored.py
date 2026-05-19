"""
Tests for modeling_selfless_refactored.py

Verifies:
1. Logic correctness under all combinations of self.training / self.calculate_likelihood
2. Equivalence: refactored(eval, calculate_likelihood=True) == original(train)

Uses model weights from output/selfless-0.6B-50BT/hf_model-final.
"""

import gc
import pytest
import torch
from safetensors.torch import load_file
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from models.modeling_model.modeling_selfless_refactored import (
    Qwen3ForCausalLM as RefactoredLM,
    Qwen3Model as RefactoredModel,
)
from models.modeling_model.modeling_selfless import (
    Qwen3ForCausalLM as OriginalLM,
    Qwen3Model as OriginalModel,
)
from utils.utils import get_selfless_mask, get_selfless_ar_mask


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PATH = "output/selfless-0.6B-50BT/hf_model-final"
SEQ_LEN = 1024
BATCH_SIZE = 2

# Tolerances for torch.allclose / assert_close
# Cross-compilation comparisons (compiled_flex_attention vs dynamic_flex_attention)
RTOL = 1e-2
ATOL = 1e-2
# Same-compilation comparisons (both compiled_flex_attention or both uncompiled)
STRICT_RTOL = 1e-4
STRICT_ATOL = 1e-4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _needs_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for flex_attention")


def make_test_v_sample(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    """Create a v_sample tensor for get_selfless_mask.

    Uses a strictly decreasing pattern (like AR mode): first token has the
    highest value, last has the lowest. This creates a causal-like mask
    where each position can only attend to earlier positions.
    """
    eps = 1e-3
    pos_idx = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(0)
    if seq_len > 1:
        v_sample = 1 - eps - (1 - 2 * eps) * pos_idx / (seq_len - 1)
    else:
        v_sample = torch.ones(1, 1, device=device) * (1 - eps)
    return v_sample.expand(batch_size, -1)


# ---------------------------------------------------------------------------
# Module-scoped fixtures (loaded once, shared across all tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def device():
    _needs_cuda()
    return torch.device("cuda")


@pytest.fixture(scope="module")
def config():
    return Qwen3Config.from_pretrained(MODEL_PATH)


@pytest.fixture(scope="module")
def state_dict(device):
    sd = load_file(f"{MODEL_PATH}/model.safetensors")
    return {k: v.to(device=device, dtype=torch.bfloat16) for k, v in sd.items()}


@pytest.fixture(scope="module")
def input_ids(device):
    """Random token IDs that avoid mask_token_id (151669) so X0 != XT."""
    return torch.randint(0, 10000, (BATCH_SIZE, SEQ_LEN), device=device, dtype=torch.long)


@pytest.fixture(scope="module")
def selfless_mask(device):
    """BlockMask created via get_selfless_mask (diffusion-style attention)."""
    v_sample = make_test_v_sample(BATCH_SIZE, SEQ_LEN, device)
    return get_selfless_mask(v_sample=v_sample, seq_len=SEQ_LEN, device=device)


@pytest.fixture(scope="module")
def ar_mask(device):
    """Strict causal BlockMask via get_selfless_ar_mask."""
    return get_selfless_ar_mask(seq_len=SEQ_LEN, B=BATCH_SIZE, device=device)


@pytest.fixture(scope="module")
def refactored_model(config, state_dict, device):
    """Refactored model loaded with saved weights, starting in eval mode."""
    model = RefactoredLM(config).to(device=device, dtype=torch.bfloat16)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Warm-up compiled attention paths (train + eval)
    _warmup_model(model, device)

    model.model.XT_input_ids = None
    return model


@pytest.fixture(scope="module")
def original_model(config, state_dict, device):
    """Original model loaded with the same saved weights."""
    model = OriginalLM(config).to(device=device, dtype=torch.bfloat16)
    model.load_state_dict(state_dict, strict=False)

    _warmup_original(model, device)

    model.model.XT_input_ids = None
    return model


def _warmup_model(model, device):
    """Warm up torch.compile caches for the refactored model."""
    warm_len = 128
    warm_input = torch.randint(0, 10000, (1, warm_len), device=device)
    v_sample = make_test_v_sample(1, warm_len, device)
    warm_mask = get_selfless_mask(v_sample=v_sample, seq_len=warm_len, device=device)
    warm_ar = get_selfless_ar_mask(seq_len=warm_len, device=device)

    model.train()
    with torch.no_grad():
        _ = model(X0_input_ids=warm_input, attention_mask=warm_mask, calculate_likelihood=True)
    model.eval()
    with torch.no_grad():
        _ = model(X0_input_ids=warm_input, attention_mask=warm_ar, calculate_likelihood=True)
        _ = model(X0_input_ids=warm_input, attention_mask=warm_ar, calculate_likelihood=False)
    torch.cuda.synchronize()


def _warmup_original(model, device):
    """Warm up torch.compile caches for the original model."""
    warm_len = 128
    warm_input = torch.randint(0, 10000, (1, warm_len), device=device)
    v_sample = make_test_v_sample(1, warm_len, device)
    warm_mask = get_selfless_mask(v_sample=v_sample, seq_len=warm_len, device=device)

    model.train()
    with torch.no_grad():
        _ = model(X0_input_ids=warm_input, attention_mask=warm_mask)
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Part 1 — Refactored model logic: training / calculate_likelihood combinations
# ---------------------------------------------------------------------------

class TestRefactoredModelLogic:
    """
    Verify that the refactored model's XT / X0 path selection behaves
    correctly across all combinations of self.training and self.calculate_likelihood.
    """

    def test_train_mode_produces_xt_output(self, refactored_model, input_ids, selfless_mask):
        """train() -> XT path active -> output should NOT match eval X0 output."""
        refactored_model.train()
        refactored_model.model.XT_input_ids = None
        with torch.no_grad():
            out_train = refactored_model(X0_input_ids=input_ids, attention_mask=selfless_mask)

        refactored_model.eval()
        refactored_model.model.XT_input_ids = None
        with torch.no_grad():
            out_eval_x0 = refactored_model(
                X0_input_ids=input_ids, attention_mask=selfless_mask, calculate_likelihood=False
            )

        # XT path (mask tokens) vs X0 path (real tokens) -> logits must differ
        allclose = torch.allclose(out_train.logits, out_eval_x0.logits, rtol=RTOL, atol=ATOL)
        assert not allclose, (
            "Train (XT-path) and eval X0-path outputs should differ, but they are allclose. "
            "max_diff="
            + str((out_train.logits.float() - out_eval_x0.logits.float()).abs().max().item())
        )

    def test_train_mode_with_calc_lik_true_still_xt(self, refactored_model, input_ids, selfless_mask):
        """train() + calculate_likelihood=True -> still XT path (training gates it)."""
        refactored_model.train()
        refactored_model.model.XT_input_ids = None
        with torch.no_grad():
            out1 = refactored_model(X0_input_ids=input_ids, attention_mask=selfless_mask)
            out2 = refactored_model(
                X0_input_ids=input_ids, attention_mask=selfless_mask, calculate_likelihood=True
            )

        torch.testing.assert_close(
            out1.logits.float(), out2.logits.float(), rtol=STRICT_RTOL, atol=STRICT_ATOL
        )

    def test_eval_calc_lik_true_uses_xt_path(self, refactored_model, input_ids, ar_mask):
        """eval() + calculate_likelihood=True -> XT path -> close to train() output.

        Uses AR mask because both train (compiled) and eval (dynamic) produce
        deterministic results with the simpler causal pattern.
        """
        refactored_model.train()
        refactored_model.model.XT_input_ids = None
        with torch.no_grad():
            out_train = refactored_model(X0_input_ids=input_ids, attention_mask=ar_mask)

        refactored_model.eval()
        refactored_model.model.XT_input_ids = None
        with torch.no_grad():
            out_eval_xt = refactored_model(
                X0_input_ids=input_ids, attention_mask=ar_mask, calculate_likelihood=True
            )

        # Both are XT-path; attention compilation differs (compiled vs dynamic)
        # but numerical result should be approximately the same
        torch.testing.assert_close(
            out_train.logits.float(),
            out_eval_xt.logits.float(),
            rtol=RTOL,
            atol=ATOL,
        )

    def test_eval_calc_lik_false_uses_x0_path(self, refactored_model, input_ids, ar_mask):
        """eval() + calculate_likelihood=False -> X0 path -> must differ from XT path."""
        refactored_model.eval()
        refactored_model.model.XT_input_ids = None
        with torch.no_grad():
            out_x0 = refactored_model(
                X0_input_ids=input_ids, attention_mask=ar_mask, calculate_likelihood=False
            )
            out_xt = refactored_model(
                X0_input_ids=input_ids, attention_mask=ar_mask, calculate_likelihood=True
            )

        allclose = torch.allclose(out_x0.logits, out_xt.logits, rtol=RTOL, atol=ATOL)
        assert not allclose, (
            "X0-path and XT-path outputs should differ. "
            "max_diff="
            + str((out_x0.logits.float() - out_xt.logits.float()).abs().max().item())
        )

    def test_xt_input_ids_rebuilt_on_shape_change(self, refactored_model, input_ids, selfless_mask, device):
        """XT_input_ids cache is rebuilt when sequence length changes."""
        refactored_model.eval()
        refactored_model.model.XT_input_ids = None

        # Forward with original shape -> caches XT_input_ids
        with torch.no_grad():
            refactored_model(
                X0_input_ids=input_ids, attention_mask=selfless_mask, calculate_likelihood=True
            )
        assert refactored_model.model.XT_input_ids.shape == input_ids.shape, (
            f"Expected {input_ids.shape}, got {refactored_model.model.XT_input_ids.shape}"
        )

        # Forward with different length -> should rebuild
        short_len = SEQ_LEN // 2
        new_ids = input_ids[:, :short_len]
        new_v_sample = make_test_v_sample(BATCH_SIZE, short_len, device)
        new_mask = get_selfless_mask(v_sample=new_v_sample, seq_len=short_len, device=device)
        with torch.no_grad():
            refactored_model(
                X0_input_ids=new_ids, attention_mask=new_mask, calculate_likelihood=True
            )
        assert refactored_model.model.XT_input_ids.shape == new_ids.shape, (
            f"Expected {new_ids.shape}, got {refactored_model.model.XT_input_ids.shape}"
        )

    def test_forward_with_labels_computes_loss(self, refactored_model, input_ids, selfless_mask):
        """When labels are provided, loss should be computed."""
        refactored_model.train()
        refactored_model.model.XT_input_ids = None
        labels = input_ids.clone()

        out = refactored_model(
            X0_input_ids=input_ids, attention_mask=selfless_mask, labels=labels
        )
        assert out.loss is not None, "Loss should not be None when labels are provided"
        assert out.loss.item() > 0, f"Loss should be > 0, got {out.loss.item()}"

    def test_eval_forward_with_labels_and_calc_lik(self, refactored_model, input_ids, selfless_mask):
        """eval + calculate_likelihood=True + labels -> loss computed."""
        refactored_model.eval()
        refactored_model.model.XT_input_ids = None
        labels = input_ids.clone()

        out = refactored_model(
            X0_input_ids=input_ids,
            attention_mask=selfless_mask,
            labels=labels,
            calculate_likelihood=True,
        )
        assert out.loss is not None, "Loss should not be None when labels are provided"
        assert out.loss.item() > 0, f"Loss should be > 0, got {out.loss.item()}"

    def test_both_mask_types_work(self, refactored_model, input_ids, selfless_mask, ar_mask):
        """Model forward should work with both selfless_mask and ar_mask."""
        refactored_model.train()
        refactored_model.model.XT_input_ids = None
        with torch.no_grad():
            out1 = refactored_model(X0_input_ids=input_ids, attention_mask=selfless_mask)
            out2 = refactored_model(X0_input_ids=input_ids, attention_mask=ar_mask)

        # Both masks produce valid outputs (different mask patterns -> different outputs)
        assert out1.logits.shape == out2.logits.shape, "Output shapes should match"


# ---------------------------------------------------------------------------
# Part 2 — Equivalence: refactored vs original
# ---------------------------------------------------------------------------

class TestEquivalenceWithOriginal:
    """
    Verify that the refactored model in eval mode + calculate_likelihood=True
    produces the same output as the original model in training mode.

    Also tests that both models in training mode give identical results
    (since they share the same compiled_flex_attention path).
    """

    def test_both_train_produce_identical_output(
        self, refactored_model, original_model, input_ids, selfless_mask
    ):
        """Both models in train() -> same compiled_flex_attention -> identical output."""
        refactored_model.train()
        refactored_model.model.XT_input_ids = None
        original_model.train()
        original_model.model.XT_input_ids = None

        with torch.no_grad():
            out_ref = refactored_model(X0_input_ids=input_ids, attention_mask=selfless_mask)
            out_orig = original_model(X0_input_ids=input_ids, attention_mask=selfless_mask)

        # Same weights, same compiled_flex_attention -> bit-identical
        max_diff = (out_ref.logits.float() - out_orig.logits.float()).abs().max().item()
        assert max_diff < STRICT_ATOL, (
            f"Both-train outputs should be identical. max_diff={max_diff}"
        )

    def test_refactored_eval_calc_lik_equals_original_train(
        self, refactored_model, original_model, input_ids, ar_mask
    ):
        """Refactored(eval, calculate_likelihood=True) matches Original(train).

        This is the key equivalence: the refactored model separates the XT-path
        gate from self.training, so eval + calculate_likelihood=True should
        reproduce the original training-mode forward pass.

        Uses AR mask for deterministic cross-compilation comparison.
        """
        refactored_model.eval()
        refactored_model.model.XT_input_ids = None
        original_model.train()
        original_model.model.XT_input_ids = None

        with torch.no_grad():
            out_ref = refactored_model(
                X0_input_ids=input_ids,
                attention_mask=ar_mask,
                calculate_likelihood=True,
            )
            out_orig = original_model(X0_input_ids=input_ids, attention_mask=ar_mask)

        torch.testing.assert_close(
            out_ref.logits.float(),
            out_orig.logits.float(),
            rtol=RTOL,
            atol=ATOL,
        )

    def test_refactored_eval_calc_lik_false_equals_original_eval(
        self, refactored_model, original_model, input_ids, ar_mask
    ):
        """Refactored(eval, calculate_likelihood=False) == Original(eval) -- both X0 path."""
        refactored_model.eval()
        refactored_model.model.XT_input_ids = None
        original_model.eval()
        original_model.model.XT_input_ids = None

        with torch.no_grad():
            out_ref = refactored_model(
                X0_input_ids=input_ids,
                attention_mask=ar_mask,
                calculate_likelihood=False,
            )
            out_orig = original_model(X0_input_ids=input_ids, attention_mask=ar_mask)

        max_diff = (out_ref.logits.float() - out_orig.logits.float()).abs().max().item()
        assert max_diff < STRICT_ATOL, (
            f"Both-eval-X0 outputs should be identical. max_diff={max_diff}"
        )

    def test_loss_equivalence_train_vs_eval_calc_lik(
        self, refactored_model, original_model, input_ids, ar_mask
    ):
        """Loss values match between refactored(eval, calc_lik=True) and original(train)."""
        labels = input_ids.clone()

        refactored_model.eval()
        refactored_model.model.XT_input_ids = None
        original_model.train()
        original_model.model.XT_input_ids = None

        with torch.no_grad():
            out_ref = refactored_model(
                X0_input_ids=input_ids,
                attention_mask=ar_mask,
                labels=labels,
                calculate_likelihood=True,
            )
            out_orig = original_model(
                X0_input_ids=input_ids, attention_mask=ar_mask, labels=labels
            )

        torch.testing.assert_close(
            torch.tensor(out_ref.loss.item()),
            torch.tensor(out_orig.loss.item()),
            rtol=RTOL,
            atol=ATOL,
        )

    def test_diffusion_mask_equivalence(
        self, refactored_model, original_model, input_ids, selfless_mask
    ):
        """Equivalence also holds with diffusion (selfless) attention masks."""
        refactored_model.eval()
        refactored_model.model.XT_input_ids = None
        original_model.train()
        original_model.model.XT_input_ids = None

        with torch.no_grad():
            out_ref = refactored_model(
                X0_input_ids=input_ids,
                attention_mask=selfless_mask,
                calculate_likelihood=True,
            )
            out_orig = original_model(X0_input_ids=input_ids, attention_mask=selfless_mask)

        torch.testing.assert_close(
            out_ref.logits.float(),
            out_orig.logits.float(),
            rtol=RTOL,
            atol=ATOL,
        )
