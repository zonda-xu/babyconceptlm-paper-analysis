"""Boundary-only inference adapted from the full-dev audit."""
import torch

def move_boundary_path(model: torch.nn.Module, device: torch.device) -> None:
    base = model.concept_gpt_bert
    base.embeddings.to(device)
    base.token_shell.to(device)
    if getattr(base, "token_dwa", None) is not None:
        base.token_dwa.to(device)
    base.boundary_scorer.to(device)
    base.eval()


@torch.inference_mode()
def count_segments(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Reproduce the eval-mode token path and deterministic mass decoder."""
    base = model.concept_gpt_bert
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    language_ids = batch["language_ids"].to(device, non_blocking=True)
    if language_ids.ndim == 1:
        language_ids = language_ids[:, None].expand_as(input_ids)

    position_ids = (torch.cumsum(attention_mask, dim=1) - 1).clamp_min(0)
    token_states = base.embeddings(input_ids, position_ids, language_ids)
    token_dwa = getattr(base, "token_dwa", None)
    dwa_states = [token_states] if token_dwa is not None else None
    for layer_index, block in enumerate(base.token_shell):
        if (
            token_dwa is not None
            and dwa_states is not None
            and base.dwa_granularity == "subblock"
        ):
            token_states = token_states + block.dropout1(
                block.attn(
                    block.norm1(token_states),
                    attention_mask=attention_mask,
                    is_causal=True,
                )
            )
            dwa_states.append(token_states)
            token_states = token_dwa(dwa_states, layer_index * 2)
            dwa_states[-1] = token_states
            token_states = token_states + block.dropout2(
                block.ffn(block.norm2(token_states))
            )
            dwa_states.append(token_states)
            token_states = token_dwa(dwa_states, layer_index * 2 + 1)
            dwa_states[-1] = token_states
        else:
            token_states = block(
                token_states,
                attention_mask=attention_mask,
                is_causal=True,
            )
            if token_dwa is not None and dwa_states is not None:
                dwa_states.append(token_states)
                token_states = token_dwa(dwa_states, layer_index)
                dwa_states[-1] = token_states

    syntax_labels = None
    if bool(getattr(model.config, "boundary_use_syntax_input", False)):
        syntax_labels = model._derive_syntax_token_labels(input_ids, attention_mask)
    scores = base.boundary_scorer(
        token_states,
        attention_mask,
        is_causal=True,
        syntax_token_labels=syntax_labels,
    )
    probs = base.segment_pooler._boundary_probabilities(scores, attention_mask)
    forced = base.segment_pooler._forced_boundaries(attention_mask, input_ids).to(
        probs.dtype
    )
    hard = base.segment_pooler._decode_boundaries_by_mass(
        probs,
        forced,
        attention_mask,
    )
    return hard.sum(dim=1).to("cpu", dtype=torch.int16)
