"""EviAlign-CXR: classifier with exact patch evidence + explanation decoder with token grounding."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoders import CrossAttnDecoder, PrefixLLMDecoder
from .encoder import PatchEncoder
from .head import EvidenceHead


class EviAlign(nn.Module):
    def __init__(self, cfg: dict, n_classes: int):
        super().__init__()
        m = cfg["model"]
        self.encoder = PatchEncoder(m["backbone"], cfg["data"]["image_size"], pretrained=m.get("pretrained", True),
                                    grad_ckpt=m.get("grad_ckpt", False), timm_kwargs=m.get("timm_kwargs"))
        self.head = EvidenceHead(self.encoder.dim, n_classes, d=m.get("head_dim", 256), heads=m.get("head_heads", 4))
        self.mem_grid = m.get("memory_grid", 16)
        # if true, no gradient from the decoder (LM, alignment, instance terms) reaches the encoder
        self.detach_memory = m.get("decoder_detach_memory", False)
        dec = dict(m.get("decoder", {"type": "none"}))
        t = dec.get("type", "none")
        tiny_tok = None
        if dec.get("tiny_config") is not None:          # CPU smoke tests only
            from tests.tiny_tokenizer import TinyTokenizer
            tiny_tok = TinyTokenizer()
            tc = dict(dec["tiny_config"])
            tc.update(vocab_size=tiny_tok.vocab_size, bos_token_id=tiny_tok.bos_token_id,
                      eos_token_id=tiny_tok.eos_token_id, pad_token_id=tiny_tok.pad_token_id)
            dec["tiny_config"] = tc
        if t == "crossattn":
            self.decoder = CrossAttnDecoder(self.encoder.dim, dec.get("lm_name", "gpt2"), dec.get("n_ground_layers", 2),
                                            pretrained=dec.get("pretrained", True), tiny_config=dec.get("tiny_config"))
        elif t == "prefix_llm":
            self.decoder = PrefixLLMDecoder(self.encoder.dim, dec["lm_name"], dec.get("n_ground_layers", 4),
                                            lora_r=dec.get("lora_r", 16), prompt=dec.get("prompt", "Findings:"),
                                            tiny_config=dec.get("tiny_config"), dtype=dec.get("dtype", "bfloat16"))
        else:
            self.decoder = None
        if tiny_tok is not None:
            self.decoder.tok = tiny_tok
            if hasattr(self.decoder, "prompt_ids"):
                self.decoder.prompt_ids = torch.tensor([tiny_tok.encode_plain("findings :")])

    @property
    def tokenizer(self):
        return None if self.decoder is None else self.decoder.tok

    def encode(self, x):
        return self.encoder(x)

    def classify(self, z):
        return self.head(z)

    def memory(self, z):
        B, M, D = z.shape
        g = int(math.isqrt(M))
        if g == self.mem_grid:
            return z
        x = z.transpose(1, 2).reshape(B, D, g, g)
        x = F.adaptive_avg_pool2d(x, self.mem_grid)
        return x.flatten(2).transpose(1, 2)

    def forward(self, x, tgt_ids=None, tgt_attn=None):
        z = self.encode(x)
        logits, phi, beta = self.classify(z)
        out = {"z": z, "logits": logits, "phi": phi, "beta": beta}
        if self.decoder is not None and tgt_ids is not None:
            mem = self.memory(z.detach() if self.detach_memory else z)
            lm, g = self.decoder(mem, tgt_ids, tgt_attn)
            out.update(lm_loss=lm, grounding=g, mem=mem)
        return out

    @torch.no_grad()
    def generate(self, x, max_new_tokens=96):
        z = self.encode(x)
        logits, phi, beta = self.classify(z)
        mem = self.memory(z)
        ids, attn = self.decoder.generate_ids(mem, max_new_tokens)
        return {"logits": logits, "phi": phi, "beta": beta, "mem": mem, "gen_ids": ids, "gen_attn": attn}
