from typing import Optional

import torch


def _apply_top_k_top_p(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    logits = logits.clone()
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        min_values = values[..., -1, None]
        logits = torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False

        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(1, sorted_indices, sorted_mask)
        logits = logits.masked_fill(mask, float("-inf"))
    return logits


def _sample_token(
    logits: torch.Tensor,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    if not do_sample:
        return torch.argmax(logits, dim=-1, keepdim=True)

    temperature = max(temperature, 1e-5)
    logits = logits / temperature
    logits = _apply_top_k_top_p(logits, top_k=top_k, top_p=top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _sampling_probs(
    logits: torch.Tensor,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    if not do_sample:
        probs = torch.zeros_like(logits)
        probs.scatter_(1, torch.argmax(logits, dim=-1, keepdim=True), 1.0)
        return probs
    temperature = max(temperature, 1e-5)
    logits = logits / temperature
    logits = _apply_top_k_top_p(logits, top_k=top_k, top_p=top_p)
    return torch.softmax(logits, dim=-1)


def _trim_past_key_values(past_key_values, keep_len: int):
    trimmed = []
    for layer_past in past_key_values:
        key, value = layer_past
        trimmed.append((key[..., :keep_len, :], value[..., :keep_len, :]))
    return tuple(trimmed)


@torch.inference_mode()
def speculative_decode(
    model,
    assistant_model,
    input_ids: torch.LongTensor,
    point_clouds: Optional[torch.FloatTensor] = None,
    max_length: int = 2048,
    eos_token_id: Optional[int] = None,
    num_assistant_tokens: int = 8,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    stopping_criteria=None,
    return_stats: bool = False,
) -> torch.LongTensor:
    if input_ids.shape[0] != 1:
        raise ValueError("Current speculative_decode only supports batch_size=1.")
    if assistant_model is None:
        raise ValueError("assistant_model must not be None for speculative decoding.")

    generated = input_ids
    stats = {
        "mode": "linear",
        "iterations": 0,
        "draft_proposed_tokens": 0,
        "accepted_draft_tokens": 0,
        "target_corrections": 0,
    }

    target_outputs = model(
        input_ids=generated,
        point_clouds=point_clouds,
        use_cache=True,
        return_dict=True,
    )
    target_past = target_outputs.past_key_values
    target_next_logits = target_outputs.logits[:, -1, :]

    draft_outputs = assistant_model(
        input_ids=generated,
        point_clouds=point_clouds,
        use_cache=True,
        return_dict=True,
    )
    draft_past = draft_outputs.past_key_values
    draft_next_logits = draft_outputs.logits[:, -1, :]

    while generated.shape[1] < max_length:
        stats["iterations"] += 1
        draft_tokens = []
        draft_token_dists = []
        for _ in range(num_assistant_tokens):
            draft_token_dists.append(
                _sampling_probs(
                    draft_next_logits,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
            )
            next_token = _sample_token(
                draft_next_logits,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            draft_tokens.append(next_token)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

            draft_step = assistant_model(
                input_ids=next_token,
                past_key_values=draft_past,
                use_cache=True,
                return_dict=True,
            )
            draft_past = draft_step.past_key_values
            draft_next_logits = draft_step.logits[:, -1, :]

        if len(draft_tokens) == 0:
            break
        draft_tokens = torch.cat(draft_tokens, dim=1)
        num_draft = draft_tokens.shape[1]
        stats["draft_proposed_tokens"] += num_draft

        verify_outputs = model(
            input_ids=draft_tokens,
            past_key_values=target_past,
            use_cache=True,
            return_dict=True,
        )
        verify_logits = verify_outputs.logits
        verify_past_full = verify_outputs.past_key_values

        accepted = 0
        corrected = None
        for idx in range(num_draft):
            if idx == 0:
                ref_logits = target_next_logits
            else:
                ref_logits = verify_logits[:, idx - 1, :]
            draft_token_id = int(draft_tokens[0, idx].item())
            if not do_sample:
                target_choice = torch.argmax(ref_logits, dim=-1, keepdim=True)
                if target_choice.item() == draft_token_id:
                    accepted += 1
                else:
                    corrected = target_choice
                    break
            else:
                p_probs = _sampling_probs(
                    ref_logits,
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                q_probs = draft_token_dists[idx]
                p_x = p_probs[0, draft_token_id]
                q_x = q_probs[0, draft_token_id].clamp_min(1e-8)
                alpha = torch.minimum(torch.ones_like(p_x), p_x / q_x)
                if torch.rand((), device=generated.device) <= alpha:
                    accepted += 1
                else:
                    residual = torch.clamp(p_probs - q_probs, min=0.0)
                    residual_sum = residual.sum(dim=-1, keepdim=True)
                    if residual_sum.item() <= 1e-8:
                        residual = p_probs
                        residual_sum = residual.sum(dim=-1, keepdim=True)
                    residual = residual / residual_sum
                    corrected = torch.multinomial(residual, num_samples=1)
                    break
        stats["accepted_draft_tokens"] += accepted

        context_len = generated.shape[1]
        if accepted < num_draft:
            if corrected is None:
                # defensive fallback, should normally be set when accepted < num_draft
                mismatch_ref_logits = target_next_logits if accepted == 0 else verify_logits[:, accepted - 1, :]
                corrected = _sample_token(
                    mismatch_ref_logits,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )

            if accepted > 0:
                generated = torch.cat([generated, draft_tokens[:, :accepted]], dim=1)
                if eos_token_id is not None and (draft_tokens[0, :accepted] == eos_token_id).any():
                    break
                if stopping_criteria is not None and any(criteria(generated, None) for criteria in stopping_criteria):
                    break
            generated = torch.cat([generated, corrected], dim=1)
            stats["target_corrections"] += 1
            if eos_token_id is not None and corrected.item() == eos_token_id:
                break
            if stopping_criteria is not None and any(criteria(generated, None) for criteria in stopping_criteria):
                break

            target_keep = context_len + accepted
            target_past = _trim_past_key_values(verify_past_full, keep_len=target_keep)
            target_step = model(
                input_ids=corrected,
                past_key_values=target_past,
                use_cache=True,
                return_dict=True,
            )
            target_past = target_step.past_key_values
            target_next_logits = target_step.logits[:, -1, :]

            draft_keep = context_len + accepted
            draft_past = _trim_past_key_values(draft_past, keep_len=draft_keep)
            draft_step = assistant_model(
                input_ids=corrected,
                past_key_values=draft_past,
                use_cache=True,
                return_dict=True,
            )
            draft_past = draft_step.past_key_values
            draft_next_logits = draft_step.logits[:, -1, :]
        else:
            generated = torch.cat([generated, draft_tokens], dim=1)
            target_past = verify_past_full
            if eos_token_id is not None and (draft_tokens[0] == eos_token_id).any():
                break
            if stopping_criteria is not None and any(criteria(generated, None) for criteria in stopping_criteria):
                break
            extra = _sample_token(
                verify_logits[:, -1, :],
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            generated = torch.cat([generated, extra], dim=1)
            if eos_token_id is not None and extra.item() == eos_token_id:
                break
            if stopping_criteria is not None and any(criteria(generated, None) for criteria in stopping_criteria):
                break

            target_step = model(
                input_ids=extra,
                past_key_values=target_past,
                use_cache=True,
                return_dict=True,
            )
            target_past = target_step.past_key_values
            target_next_logits = target_step.logits[:, -1, :]

            draft_step = assistant_model(
                input_ids=extra,
                past_key_values=draft_past,
                use_cache=True,
                return_dict=True,
            )
            draft_past = draft_step.past_key_values
            draft_next_logits = draft_step.logits[:, -1, :]

        if generated.shape[1] >= max_length:
            break

    if return_stats:
        proposed = max(stats["draft_proposed_tokens"], 1)
        stats["acceptance_rate"] = stats["accepted_draft_tokens"] / proposed
        return generated[:, :max_length], stats
    return generated[:, :max_length]


def _top_tokens_for_tree(logits: torch.Tensor, branching_factor: int, temperature: float) -> torch.Tensor:
    temperature = max(temperature, 1e-5)
    probs = torch.softmax(logits / temperature, dim=-1)
    _, top_idx = torch.topk(probs, k=branching_factor, dim=-1)
    return top_idx


@torch.inference_mode()
def tree_speculative_decode(
    model,
    assistant_model,
    input_ids: torch.LongTensor,
    point_clouds: Optional[torch.FloatTensor] = None,
    max_length: int = 2048,
    eos_token_id: Optional[int] = None,
    tree_depth: int = 4,
    tree_branching_factor: int = 2,
    tree_max_paths: int = 16,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    stopping_criteria=None,
    return_stats: bool = False,
) -> torch.LongTensor:
    if input_ids.shape[0] != 1:
        raise ValueError("Current tree_speculative_decode only supports batch_size=1.")
    if assistant_model is None:
        raise ValueError("assistant_model must not be None for tree speculative decoding.")
    if tree_depth <= 0 or tree_branching_factor <= 0:
        raise ValueError("tree_depth and tree_branching_factor must be > 0.")

    generated = input_ids
    stats = {
        "mode": "tree",
        "iterations": 0,
        "draft_proposed_tokens": 0,
        "accepted_draft_tokens": 0,
        "target_corrections": 0,
    }

    def recompute_states(cur_generated: torch.Tensor):
        target_out = model(
            input_ids=cur_generated,
            point_clouds=point_clouds,
            use_cache=True,
            return_dict=True,
        )
        draft_out = assistant_model(
            input_ids=cur_generated,
            point_clouds=point_clouds,
            use_cache=True,
            return_dict=True,
        )
        return target_out, draft_out

    target_outputs, draft_outputs = recompute_states(generated)
    target_next_logits = target_outputs.logits[:, -1, :]

    while generated.shape[1] < max_length:
        stats["iterations"] += 1
        # build a tree of candidate paths with assistant model (beam-style expansion)
        beams = [{
            "tokens": [],
            "past": draft_outputs.past_key_values,
            "logits": draft_outputs.logits[:, -1, :],
            "score": 0.0,
        }]
        completed = []
        for _ in range(tree_depth):
            new_beams = []
            for beam in beams:
                top_tokens = _top_tokens_for_tree(
                    beam["logits"],
                    branching_factor=tree_branching_factor,
                    temperature=temperature,
                )
                for tok in top_tokens[0]:
                    tok = tok.view(1, 1)
                    step_out = assistant_model(
                        input_ids=tok,
                        past_key_values=beam["past"],
                        use_cache=True,
                        return_dict=True,
                    )
                    token_logprob = torch.log_softmax(beam["logits"] / max(temperature, 1e-5), dim=-1)[0, tok.item()].item()
                    new_tokens = beam["tokens"] + [tok.item()]
                    candidate = {
                        "tokens": new_tokens,
                        "past": step_out.past_key_values,
                        "logits": step_out.logits[:, -1, :],
                        "score": beam["score"] + token_logprob,
                    }
                    new_beams.append(candidate)
                    stats["draft_proposed_tokens"] += 1
                    if eos_token_id is not None and tok.item() == eos_token_id:
                        completed.append(candidate)
            if len(new_beams) == 0:
                break
            new_beams = sorted(new_beams, key=lambda x: x["score"], reverse=True)[:tree_max_paths]
            beams = new_beams
            completed.extend(beams)

        candidates = sorted(completed if len(completed) > 0 else beams, key=lambda x: x["score"], reverse=True)[:tree_max_paths]
        if len(candidates) == 0:
            break

        # verify all candidate paths with target model and pick the one with longest accepted prefix
        best = None
        for cand in candidates:
            cand_tokens = torch.tensor(cand["tokens"], dtype=torch.long, device=generated.device).unsqueeze(0)
            verify_out = model(
                input_ids=cand_tokens,
                past_key_values=target_outputs.past_key_values,
                use_cache=True,
                return_dict=True,
            )
            verify_logits = verify_out.logits

            accepted = 0
            corrected = None
            for idx in range(cand_tokens.shape[1]):
                if idx == 0:
                    ref_logits = target_next_logits
                else:
                    ref_logits = verify_logits[:, idx - 1, :]
                target_choice = _sample_token(
                    ref_logits,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                if target_choice.item() == cand_tokens[0, idx].item():
                    accepted += 1
                else:
                    corrected = target_choice
                    break

            current = {
                "accepted": accepted,
                "cand_tokens": cand_tokens,
                "verify_logits": verify_logits,
                "score": cand["score"],
                "corrected": corrected,
            }
            if best is None or current["accepted"] > best["accepted"] or (
                current["accepted"] == best["accepted"] and current["score"] > best["score"]
            ):
                best = current

        if best is None:
            break

        cand_tokens = best["cand_tokens"]
        accepted = best["accepted"]
        stats["accepted_draft_tokens"] += accepted
        verify_logits = best["verify_logits"]

        if accepted == 0:
            corrected = best.get("corrected", None)
            if corrected is None:
                corrected = _sample_token(
                    target_next_logits,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
            generated = torch.cat([generated, corrected], dim=1)
            stats["target_corrections"] += 1
        elif accepted < cand_tokens.shape[1]:
            generated = torch.cat([generated, cand_tokens[:, :accepted]], dim=1)
            if eos_token_id is not None and (cand_tokens[0, :accepted] == eos_token_id).any():
                break
            if stopping_criteria is not None and any(criteria(generated, None) for criteria in stopping_criteria):
                break
            corrected = best.get("corrected", None)
            if corrected is None:
                mismatch_ref_logits = verify_logits[:, accepted - 1, :]
                corrected = _sample_token(
                    mismatch_ref_logits,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
            generated = torch.cat([generated, corrected], dim=1)
            stats["target_corrections"] += 1
        else:
            generated = torch.cat([generated, cand_tokens], dim=1)
            if eos_token_id is not None and (cand_tokens[0] == eos_token_id).any():
                break
            if stopping_criteria is not None and any(criteria(generated, None) for criteria in stopping_criteria):
                break
            extra = _sample_token(
                verify_logits[:, -1, :],
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            generated = torch.cat([generated, extra], dim=1)

        if eos_token_id is not None and generated[0, -1].item() == eos_token_id:
            break
        if stopping_criteria is not None and any(criteria(generated, None) for criteria in stopping_criteria):
            break
        if generated.shape[1] >= max_length:
            break

        target_outputs, draft_outputs = recompute_states(generated)
        target_next_logits = target_outputs.logits[:, -1, :]

    if return_stats:
        proposed = max(stats["draft_proposed_tokens"], 1)
        stats["acceptance_rate"] = stats["accepted_draft_tokens"] / proposed
        return generated[:, :max_length], stats
    return generated[:, :max_length]
