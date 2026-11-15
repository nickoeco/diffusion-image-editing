from types import SimpleNamespace

import pytest
import torch
from torch import nn

from diffusion_editor.models.text_encoder import TextEncoderWrapper


class TinyTokenizer:
    model_max_length = 6

    def __call__(self, prompts, padding=False, truncation=False, max_length=None, return_tensors=None):
        rows = [[1] + [ord(c) % 100 + 3 for c in p] + [2] for p in prompts]
        if truncation:
            rows = [r[:max_length] for r in rows]
        if padding:
            masks = [[1] * len(r) + [0] * (max_length - len(r)) for r in rows]
            rows = [r + [0] * (max_length - len(r)) for r in rows]
            return {"input_ids": torch.tensor(rows), "attention_mask": torch.tensor(masks)}
        return {"input_ids": rows}


class TinyCLIP(nn.Module):
    def __init__(self, attention=False):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(use_attention_mask=attention, hidden_size=3)

    def forward(self, input_ids, **kwargs):
        self.last_ids = input_ids
        self.last_kwargs = kwargs
        hidden = input_ids.to(dtype=self.anchor.dtype).unsqueeze(-1).expand(-1, -1, 3)
        return SimpleNamespace(last_hidden_state=hidden)


def test_batch_order_and_broadcast():
    model = TinyCLIP()
    wrapper = TextEncoderWrapper(TinyTokenizer(), model)
    pair = wrapper.encode_prompt_pair(["cat", "dog"])
    assert pair.shape == (4, 6, 3)
    torch.testing.assert_close(pair[:2], wrapper.encode_prompt(["", ""]))
    torch.testing.assert_close(pair[2:], wrapper.encode_prompt(["cat", "dog"]))
    negatives = wrapper.encode_prompt_pair(["cat", "dog"], ["no", "bad"])
    torch.testing.assert_close(negatives[:2], wrapper.encode_prompt(["no", "bad"]))
    torch.testing.assert_close(wrapper.encode_prompt_pair(["a", "b"], "no")[:2],
                               wrapper.encode_prompt(["no", "no"]))
    assert not model.training and not model.anchor.requires_grad


@pytest.mark.parametrize("attention", [True, False])
def test_attention_and_determinism(attention):
    model = TinyCLIP(attention)
    wrapper = TextEncoderWrapper(TinyTokenizer(), model)
    first = wrapper.encode_prompt("cat")
    second = wrapper.encode_prompt("cat")
    assert ("attention_mask" in model.last_kwargs) is attention
    assert model.last_ids.dtype == torch.int64
    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize("bad", [[], ["cat", 5], 3, None])
def test_invalid_prompt(bad):
    with pytest.raises(ValueError):
        TextEncoderWrapper(TinyTokenizer(), TinyCLIP()).encode_prompt(bad)


def test_negative_batch_mismatch():
    with pytest.raises(ValueError, match="batch length"):
        TextEncoderWrapper(TinyTokenizer(), TinyCLIP()).encode_prompt_pair(["a", "b"], ["no"])


def test_warns_on_truncation():
    with pytest.warns(UserWarning, match="truncated"):
        result = TextEncoderWrapper(TinyTokenizer(), TinyCLIP()).encode_prompt("long prompt")
    assert result.shape == (1, 6, 3)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_clip_cuda_fp16_and_token_device():
    model = TinyCLIP(attention=True).cuda().half()
    output = TextEncoderWrapper(TinyTokenizer(), model).encode_prompt_pair(["cat", "dog"])
    assert output.device.type == "cuda" and output.dtype == torch.float16
    assert model.last_ids.device.type == "cuda"
    assert model.last_kwargs["attention_mask"].device.type == "cuda"
