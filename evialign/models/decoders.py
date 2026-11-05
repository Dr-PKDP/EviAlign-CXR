"""Explanation decoders that expose token-to-patch grounding.

Both decoders implement
    forward(memory, tgt_ids, tgt_attn) -> (lm_loss, grounding)   grounding: (B, T, P)
    generate(memory, max_new_tokens)   -> list[str]
    ground(memory, ids, attn)          -> (B, T, P)   teacher-forced grounding of any text
where grounding[b, j, p] is the attention mass that the step *producing* target token j places on
visual token p, averaged over heads and over the last ``n_ground_layers`` layers and renormalised
over the P visual tokens.

CrossAttnDecoder : GPT-2-style decoder with explicit cross-attention to visual tokens.
PrefixLLMDecoder : LLaVA-style VLM; projected visual tokens are a soft prefix of a causal LLM
                   (LoRA-adapted); grounding is the text-to-prefix block of self-attention.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _renorm(a: torch.Tensor) -> torch.Tensor:
    return a / a.sum(-1, keepdim=True).clamp_min(1e-8)


class CrossAttnDecoder(nn.Module):
    def __init__(self, mem_dim: int, lm_name: str = "gpt2", n_ground_layers: int = 2,
                 pretrained: bool = True, tiny_config: dict | None = None):
        super().__init__()
        from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
        if tiny_config is not None:
            cfg = GPT2Config(add_cross_attention=True, **tiny_config)
            cfg._attn_implementation = "eager"
            self.lm = GPT2LMHeadModel(cfg)
            self.tok = None
        else:
            if pretrained:
                self.lm = GPT2LMHeadModel.from_pretrained(lm_name, add_cross_attention=True, attn_implementation="eager")
            else:
                cfg = GPT2Config.from_pretrained(lm_name, add_cross_attention=True)
                cfg._attn_implementation = "eager"
                self.lm = GPT2LMHeadModel(cfg)
            self.tok = AutoTokenizer.from_pretrained(lm_name)
            self.tok.pad_token = self.tok.eos_token
        self.hid = self.lm.config.n_embd
        self.proj = nn.Sequential(nn.Linear(mem_dim, self.hid), nn.GELU(), nn.Linear(self.hid, self.hid))
        self.bos = self.lm.config.bos_token_id
        self.eos = self.lm.config.eos_token_id
        self.ng = n_ground_layers

    def _run(self, memory, ids, attn):
        B = ids.shape[0]
        bos = torch.full((B, 1), self.bos, dtype=ids.dtype, device=ids.device)
        inp = torch.cat([bos, ids], 1)
        am = torch.cat([torch.ones_like(bos), attn], 1)
        out = self.lm(input_ids=inp, attention_mask=am, encoder_hidden_states=self.proj(memory),
                      output_attentions=True, return_dict=True)
        ca = torch.stack(out.cross_attentions[-self.ng:], 0).mean((0, 2))   # B, T+1, P
        return out, inp, am, _renorm(ca[:, :-1, :])                          # query j predicts token j

    def forward(self, memory, tgt_ids, tgt_attn):
        out, inp, am, g = self._run(memory, tgt_ids, tgt_attn)
        logits = out.logits[:, :-1, :]
        loss_tok = nn.functional.cross_entropy(logits.transpose(1, 2), tgt_ids, reduction="none")
        lm_loss = (loss_tok * tgt_attn).sum(1) / tgt_attn.sum(1).clamp_min(1)   # per-sample
        return lm_loss, g

    def ground(self, memory, ids, attn):
        return self._run(memory, ids, attn)[3]

    def token_logprob(self, memory, ids, attn):
        out, *_ = self._run(memory, ids, attn)
        lp = out.logits[:, :-1, :].log_softmax(-1).gather(-1, ids[..., None])[..., 0]
        return lp * attn

    @torch.no_grad()
    def generate_ids(self, memory, max_new_tokens: int = 96, force_ids=None):
        B = memory.shape[0]
        mem = self.proj(memory)
        ids = torch.full((B, 1), self.bos, dtype=torch.long, device=memory.device)
        if force_ids:
            ids = torch.cat([ids, torch.tensor(force_ids, dtype=torch.long, device=memory.device)[None].expand(B, -1)], 1)
        done = torch.zeros(B, dtype=torch.bool, device=memory.device)
        past = None
        for _ in range(max_new_tokens):
            out = self.lm(input_ids=ids[:, -1:] if past is not None else ids, encoder_hidden_states=mem,
                          past_key_values=past, use_cache=True, return_dict=True)
            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, self.eos), nxt)
            ids = torch.cat([ids, nxt[:, None]], 1)
            done |= nxt == self.eos
            if done.all():
                break
        gen = ids[:, 1:]
        attn = torch.ones_like(gen)
        for b in range(B):   # keep tokens up to and including the first EOS
            hit = (gen[b] == self.eos).nonzero()
            if len(hit):
                attn[b, hit[0, 0] + 1:] = 0
        return gen, attn


class PrefixLLMDecoder(nn.Module):
    def __init__(self, mem_dim: int, lm_name: str, n_ground_layers: int = 4, lora_r: int = 16,
                 prompt: str = "Findings:", tiny_config: dict | None = None, dtype: str = "bfloat16"):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if tiny_config is not None:
            from transformers import LlamaConfig, LlamaForCausalLM
            cfg = LlamaConfig(**tiny_config)
            cfg._attn_implementation = "eager"
            self.lm = LlamaForCausalLM(cfg)
            self.tok = None
            self.prompt_ids = torch.tensor([[1, 2, 3]])
        else:
            self.lm = AutoModelForCausalLM.from_pretrained(lm_name, attn_implementation="eager",
                                                           torch_dtype=getattr(torch, dtype))
            self.tok = AutoTokenizer.from_pretrained(lm_name)
            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token
            self.prompt_ids = self.tok(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
            if lora_r > 0:
                from peft import LoraConfig, get_peft_model
                self.lm = get_peft_model(self.lm, LoraConfig(
                    r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        self.hid = self.lm.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(mem_dim, self.hid), nn.GELU(), nn.Linear(self.hid, self.hid))
        self.eos = self.lm.config.eos_token_id if self.tok is None else self.tok.eos_token_id
        self.ng = n_ground_layers

    def _emb(self, ids):
        return self.lm.get_input_embeddings()(ids)

    def _prefix(self, memory):
        B, P, _ = memory.shape
        vis = self.proj(memory).to(self._emb(self.prompt_ids.to(memory.device)).dtype)
        pr = self._emb(self.prompt_ids.to(memory.device)).expand(B, -1, -1)
        return torch.cat([vis, pr], 1), P

    def _run(self, memory, ids, attn):
        pre, P = self._prefix(memory)
        L0 = pre.shape[1]
        emb = torch.cat([pre, self._emb(ids)], 1)
        am = torch.cat([torch.ones(ids.shape[0], L0, dtype=attn.dtype, device=attn.device), attn], 1)
        out = self.lm(inputs_embeds=emb, attention_mask=am, output_attentions=True, return_dict=True)
        T = ids.shape[1]
        rows = slice(L0 - 1, L0 - 1 + T)                       # position that predicts target token j
        att = torch.stack([a[:, :, rows, :P] for a in out.attentions[-self.ng:]], 0).mean((0, 2)).float()
        logits = out.logits[:, rows, :].float()
        return logits, _renorm(att)

    def forward(self, memory, tgt_ids, tgt_attn):
        logits, g = self._run(memory, tgt_ids, tgt_attn)
        loss_tok = nn.functional.cross_entropy(logits.transpose(1, 2), tgt_ids, reduction="none")
        return (loss_tok * tgt_attn).sum(1) / tgt_attn.sum(1).clamp_min(1), g

    def ground(self, memory, ids, attn):
        return self._run(memory, ids, attn)[1]

    def token_logprob(self, memory, ids, attn):
        logits, _ = self._run(memory, ids, attn)
        return logits.log_softmax(-1).gather(-1, ids[..., None])[..., 0] * attn

    @torch.no_grad()
    def generate_ids(self, memory, max_new_tokens: int = 96, force_ids=None):
        """Greedy decoding. force_ids: optional token ids that open the sentence (finding-conditioned decoding); they are
        returned as the first tokens of the output and the model continues from them."""
        pre, _ = self._prefix(memory)
        B = pre.shape[0]
        forced = []
        if force_ids:
            f = torch.tensor(force_ids, dtype=torch.long, device=pre.device)[None].expand(B, -1)
            pre = torch.cat([pre, self._emb(f)], 1)
            forced = [f[:, j] for j in range(f.shape[1])]
        am = torch.ones(B, pre.shape[1], dtype=torch.long, device=pre.device)
        out = self.lm(inputs_embeds=pre, attention_mask=am, use_cache=True, return_dict=True)
        past, nxt = out.past_key_values, out.logits[:, -1].argmax(-1)
        toks, done = forced + [nxt], nxt == self.eos
        for _ in range(max_new_tokens - 1):
            if done.all():
                break
            am = torch.cat([am, torch.ones(B, 1, dtype=am.dtype, device=am.device)], 1)
            out = self.lm(inputs_embeds=self._emb(nxt[:, None]), attention_mask=am, past_key_values=past,
                          use_cache=True, return_dict=True)
            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, self.eos), nxt)
            toks.append(nxt)
            done |= nxt == self.eos
        gen = torch.stack(toks, 1)
        attn = torch.ones_like(gen)
        for b in range(B):
            hit = (gen[b] == self.eos).nonzero()
            if len(hit):
                attn[b, hit[0, 0] + 1:] = 0
        return gen, attn
