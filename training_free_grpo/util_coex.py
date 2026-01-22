import os
import math
import json
import random
import re
import glob

def sample_experiences_grpo(experiences, experience_probs, k, rng=random):
    if k <= 0 or not experiences:
        return {}

    ids = list(experiences.keys())
    weights = [float(experience_probs.get(i, 0.5)) for i in ids]

    total = sum(max(w, 0.0) for w in weights)
    if total <= 0:
        probs = [1.0 / len(ids)] * len(ids)
    else:
        probs = [max(w, 0.0) / total for w in weights]

    # without-replacement
    selected_ids = []
    ids_pool = ids[:]
    probs_pool = probs[:]
    k = min(k, len(ids_pool))

    for _ in range(k):
        r = rng.random()
        cum = 0.0
        chosen_idx = len(probs_pool) - 1
        for idx, p in enumerate(probs_pool):
            cum += p
            if r <= cum:
                chosen_idx = idx
                break

        chosen_id = ids_pool.pop(chosen_idx)
        probs_pool.pop(chosen_idx)
        selected_ids.append(chosen_id)

        rem = sum(probs_pool)
        if rem > 0:
            probs_pool = [p / rem for p in probs_pool]

    return {i: experiences[i] for i in selected_ids}


def sample_experiences(experiences, experience_probs, k, rng):
    if k <= 0 or not experiences:
        return {}

    ids = list(experiences.keys())
    probs = [experience_probs[i] for i in ids]

    total = sum(probs)
    if total <= 0:
        # fallback to uniform
        probs = [1.0 / len(probs)] * len(probs)
    else:
        probs = [p / total for p in probs]

    selected_ids = []

    ids_pool = ids[:]
    probs_pool = probs[:]

    k = min(k, len(ids_pool))

    for _ in range(k):
        r = rng.random()
        cum = 0.0
        for idx, p in enumerate(probs_pool):
            cum += p
            if r <= cum:
                chosen_id = ids_pool.pop(idx)
                probs_pool.pop(idx)
                selected_ids.append(chosen_id)
                break

        # renormalize remaining probs
        remaining_total = sum(probs_pool)
        if remaining_total > 0:
            probs_pool = [p / remaining_total for p in probs_pool]

    return {i: experiences[i] for i in selected_ids}


def sample_experiences_bernoulli(experiences, q=0.3, k=None):
    rng = random

    selected = []
    for eid, text in experiences.items():
        if rng.random() < q:
            selected.append((eid, text))

    if k is not None and len(selected) > k:
        selected = rng.sample(selected, k)

    return dict(selected)


def sample_experiences_uniform_k(experiences, k):
    rng = random

    if not experiences or k <= 0:
        return {}
    
    items = list(experiences.items())
    k = min(k, len(items))

    selected = rng.sample(items, k)
    return dict(selected)


def accumulate_experience_scores_from_rollouts(problem_results, failure_discount=0.3):
    scores = {}
    for item in problem_results:
        rollouts = item.get("rollouts", [])
        for r in rollouts:
            exp_ids = r.get("sampled_experience_ids", [])
            if not exp_ids:
                continue

            success = 1 if r.get("reward", 0.0) else 0
            denom = float(len(exp_ids))

            if success:
                delta = 1.0 / denom
            else:
                delta = -failure_discount / denom
            for exp_id in exp_ids:
                scores[exp_id] = scores.get(exp_id, 0.0) + delta
    return scores


def load_and_init_experience_probs(experiences, path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            experience_probs = json.load(f)
    else:
        experience_probs = {}

    for exp_id in experiences.keys():
        if exp_id not in experience_probs:
            experience_probs[exp_id] = 0.5

    experience_probs = {eid: experience_probs[eid] for eid in experiences.keys()}

    with open(path, "w", encoding="utf-8") as f:
        json.dump(experience_probs, f, indent=2, ensure_ascii=False)

    return experience_probs


def update_experience_probs(old_probs, scores, eta=10, min_prob=1e-6,):
    new_probs = {}

    for exp_id, p in old_probs.items():
        s = scores.get(exp_id, 0.0)
        new_p = p * math.exp(eta * s)
        new_probs[exp_id] = max(new_p, min_prob)

    total = sum(new_probs.values())
    for exp_id in new_probs:
        new_probs[exp_id] /= total

    return new_probs


def _safe_mean_std(vals):
    if not vals:
        return 0.0, 0.0
    m = sum(vals) / float(len(vals))
    var = sum((v - m) * (v - m) for v in vals) / float(len(vals))
    return m, var ** 0.5


def compute_pi_theta_from_critique(problem_results):
    cnt = {}
    succ = {}
    for item in problem_results:
        for r in item.get("rollouts", []):
            exp_ids = r.get("sampled_experience_ids", []) or r.get("sampled_experience_id", [])
            if not exp_ids:
                continue
            reward = 1 if r.get("reward", 0.0) else 0
            for eid in exp_ids:
                cnt[eid] = cnt.get(eid, 0) + 1
                succ[eid] = succ.get(eid, 0) + reward

    pi_theta = {}
    for eid, c in cnt.items():
        pi_theta[eid] = (succ.get(eid, 0) / float(c)) if c > 0 else 0.0
    return pi_theta, cnt, succ


def compute_advantage_from_pi_theta(pi_theta, eps_adv=1e-8):
    vals = list(pi_theta.values())
    m, s = _safe_mean_std(vals)
    A = {}
    for eid, r in pi_theta.items():
        A[eid] = (r - m) / (s + eps_adv)
    return A, m, s


def clip_ratio_update(pi_old, pi_theta, eps_clip=0.2, min_denom=1e-8):
    pi_old = float(pi_old)
    pi_theta = float(pi_theta)

    denom = pi_old if abs(pi_old) > min_denom else min_denom
    ratio = pi_theta / denom
    clipped_ratio = max(1.0 - eps_clip, min(1.0 + eps_clip, ratio))

    pi_new = pi_old * clipped_ratio
    pi_new = max(0.0, min(1.0, pi_new))

    return pi_new, ratio, clipped_ratio


def parse_revision_plan_only(batch_update_result):
    plan = batch_update_result.get("revision_plan", []) or []
    out = []

    for it in plan:
        opt = it.get("option")

        sources = []
        if isinstance(it.get("merged_from"), list):
            sources = [s for s in it["merged_from"] if isinstance(s, str)]
        elif isinstance(it.get("modified_from"), str):
            sources = [it["modified_from"]]
        elif isinstance(it.get("sources"), list):
            sources = [s for s in it["sources"] if isinstance(s, str)]

        exp_text = it.get("experience")
        exp_text = exp_text if isinstance(exp_text, str) else None

        out.append({
            "option": opt,
            "experience_text": exp_text,
            "sources": sources,
        })

    return out


def load_step_experiences_and_probs(experiment_dir, step_idx):
    step_dir = os.path.join(experiment_dir, f"step_{step_idx}")
    exp_path = os.path.join(step_dir, "experiences.json")
    prob_path = os.path.join(step_dir, "experience_probs.json")
    if not (os.path.exists(exp_path) and os.path.exists(prob_path)):
        return None, None
    with open(exp_path, "r", encoding="utf-8") as f:
        exps = json.load(f)
    with open(prob_path, "r", encoding="utf-8") as f:
        probs = json.load(f)
    return exps, probs


def find_pi_old_by_text_history(experiment_dir, cur_step, target_text):
    for s in range(cur_step, -1, -1):
        exps, probs = load_step_experiences_and_probs(experiment_dir, s)
        if not exps or not probs:
            continue
        for eid, txt in exps.items():
            if txt == target_text and eid in probs:
                return probs[eid], {"found_step": s, "found_id": eid, "reason": "text_match_history"}
    return 0.5, {"found_step": None, "found_id": None, "reason": "default_0.5_no_history"}

def compute_next_experience_probs(
    experiment_dir,
    cur_step,
    cur_experiences,
    cur_probs,
    problem_results,
    batch_update_result,
    next_experiences,
    eps_clip=0.2,
    min_denom=1e-8,
):
    def clamp01(x: float) -> float:
        x = float(x)
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

    def _safe_load_json(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def compute_pi_theta_from_critique_local(problem_results_):
        cnt_ = {}
        succ_ = {}
        for item in problem_results_ or []:
            for r in (item.get("rollouts", []) or []):
                exp_ids = r.get("sampled_experience_ids", None)
                if exp_ids is None:
                    exp_ids = r.get("sampled_experience_id", None)
                if not exp_ids:
                    continue
                if isinstance(exp_ids, str):
                    exp_ids = [exp_ids]
                reward = 1 if r.get("reward", 0.0) else 0
                for eid in exp_ids:
                    if not isinstance(eid, str):
                        continue
                    cnt_[eid] = cnt_.get(eid, 0) + 1
                    succ_[eid] = succ_.get(eid, 0) + reward

        pi_theta_ = {}
        for eid, c in cnt_.items():
            pi_theta_[eid] = (succ_.get(eid, 0) / float(c)) if c > 0 else 0.0
        return pi_theta_, cnt_, succ_

    def parse_revision_plan_only_local(batch_update_result_):
        plan = (batch_update_result_ or {}).get("revision_plan", []) or []
        out = []
        for it in plan:
            if not isinstance(it, dict):
                continue
            opt = it.get("option")
            sources = []
            if isinstance(it.get("merged_from"), list):
                sources = [s for s in it["merged_from"] if isinstance(s, str)]
            elif isinstance(it.get("modified_from"), str):
                sources = [it.get("modified_from")]
            elif isinstance(it.get("sources"), list):
                sources = [s for s in it.get("sources", []) if isinstance(s, str)]

            exp_text = it.get("experience")
            exp_text = exp_text if isinstance(exp_text, str) else None

            out.append(
                {
                    "option": opt if isinstance(opt, str) else None,
                    "experience_text": exp_text,
                    "sources": sources,
                }
            )
        return out

    def load_step_experiences_and_probs_local(experiment_dir_, step_idx_):
        step_dir = os.path.join(experiment_dir_, f"step_{step_idx_}")
        exp_path = os.path.join(step_dir, "experiences.json")
        prob_path = os.path.join(step_dir, "experience_probs.json")
        if not (os.path.exists(exp_path) and os.path.exists(prob_path)):
            return None, None
        exps_ = _safe_load_json(exp_path)
        probs_ = _safe_load_json(prob_path)
        return exps_, probs_

    def find_pi_old_by_text_history_local(experiment_dir_, cur_step_, target_text_):
        for s in range(cur_step_, -1, -1):
            exps_, probs_ = load_step_experiences_and_probs_local(experiment_dir_, s)
            if not exps_ or not probs_:
                continue
            for eid, txt in (exps_ or {}).items():
                if txt == target_text_ and eid in probs_:
                    return float(probs_[eid]), {
                        "found_step": s,
                        "found_id": eid,
                        "reason": "text_match_history",
                    }
        return 0.5, {
            "found_step": None,
            "found_id": None,
            "reason": "default_0.5_no_history",
        }

    def _get_most_recent_pi_old_by_id(experiment_dir_, cur_step_, sid_):
        for s in range(cur_step_, -1, -1):
            _exps, _probs = load_step_experiences_and_probs_local(experiment_dir_, s)
            if _probs and (sid_ in _probs):
                return float(_probs[sid_]), f"history_step_{s}"
        return 0.5, "default_0.5_missing_src"

    def clip_ratio_update_local(pi_old_, pi_theta_, eps_clip_=0.2, min_denom_=1e-8):
        pi_old_ = float(pi_old_)
        pi_theta_ = float(pi_theta_)
        denom = pi_old_ if abs(pi_old_) > min_denom_ else min_denom_
        ratio_ = pi_theta_ / denom
        clipped_ratio_ = max(1.0 - eps_clip_, min(1.0 + eps_clip_, ratio_))
        pi_new_ = clamp01(pi_old_ * clipped_ratio_)
        return pi_new_, ratio_, clipped_ratio_

    def _find_plan_by_text(plan_entries_, target_text_):
        if not target_text_:
            return None
        for e in plan_entries_:
            if e.get("experience_text") == target_text_:
                return e
        return None

    # ---------- DEBUG
    debug = {
        "cur_step": cur_step,
        "pi_theta": {},                 # per eid: {pi_theta,count,success}
        "pi_old_used_for_update": {},   # per eid
        "pi_new": {},                   # per eid: update details
        "inheritance_next_step": {},    # per new_eid: inheritance details
        "meta": {
            "update_rule": "pi_new = pi_old * clip(pi_theta/pi_old, 1-eps, 1+eps)",
            "eps_clip": float(eps_clip),
            "min_denom": float(min_denom),
            "inherit_modify_uses": "MOST_RECENT_HISTORY_pi_old (NOT pi_new_cur)",
        },
    }

    # ---------- (A) pi_theta from critique
    pi_theta, cnt, succ = compute_pi_theta_from_critique_local(problem_results)
    debug["pi_theta"] = {
        eid: {
            "pi_theta": float(pi_theta.get(eid, 0.0)),
            "count": int(cnt.get(eid, 0)),
            "success": int(succ.get(eid, 0)),
        }
        for eid in pi_theta.keys()
    }

    # ---------- (A) update current step probs (pi_new_cur)
    pi_new_cur = {}
    for eid in (cur_experiences or {}).keys():
        pi_old = float((cur_probs or {}).get(eid, 0.5))
        pi_t = float(pi_theta.get(eid, 0.0))  # not observed -> 0.0

        pi_new, ratio, clipped_ratio = clip_ratio_update_local(
            pi_old_=pi_old,
            pi_theta_=pi_t,
            eps_clip_=eps_clip,
            min_denom_=min_denom,
        )
        pi_new_cur[eid] = pi_new

        debug["pi_old_used_for_update"][eid] = float(pi_old)
        debug["pi_new"][eid] = {
            "pi_old": float(pi_old),
            "pi_theta": float(pi_t),
            "ratio": float(ratio),
            "clipped_ratio": float(clipped_ratio),
            "pi_new": float(pi_new),
        }

    # ---------- (B) next step inheritance based on revision_plan only
    plan_entries = parse_revision_plan_only_local(batch_update_result) or []
    next_probs = {}

    for new_eid, new_text in (next_experiences or {}).items():
        plan_entry = _find_plan_by_text(plan_entries, new_text)
        plan_hit = plan_entry is not None
        plan_option = plan_entry.get("option") if plan_hit else None
        plan_sources = plan_entry.get("sources") if plan_hit else []

        reason = None
        info = {
            "plan_hit": bool(plan_hit),
            "plan_option": plan_option,
            "plan_sources": plan_sources,
            "matched_text": new_text if plan_hit else None,
        }

        if plan_hit and plan_sources:
            vals = []
            src_detail = []

            for sid in plan_sources:
                if plan_option == "modify":
                    v, src_from = _get_most_recent_pi_old_by_id(experiment_dir, cur_step, sid)
                    vals.append(float(v))
                    src_detail.append({"sid": sid, "from": src_from, "pi_used": float(v)})
                else:
                    if sid in pi_new_cur:
                        v = float(pi_new_cur[sid])
                        vals.append(v)
                        src_detail.append({"sid": sid, "from": "pi_new_cur", "pi_used": v})
                    else:
                        v, src_from = _get_most_recent_pi_old_by_id(experiment_dir, cur_step, sid)
                        vals.append(float(v))
                        src_detail.append({"sid": sid, "from": src_from, "pi_used": float(v)})

            if (plan_option == "modify") and (len(vals) == 1):
                next_probs[new_eid] = float(vals[0])
                reason = "revision_plan_modify_inherit_most_recent_pi_old"
            else:
                next_probs[new_eid] = float(sum(vals) / float(len(vals))) if vals else 0.5
                reason = "revision_plan_sources_avg"

            info["source_detail"] = src_detail

            if new_eid in pi_new_cur:
                info["note"] = "plan overrides same_id"
                info["same_id_pi_new"] = float(pi_new_cur[new_eid])

        elif plan_hit and not plan_sources:
            p, meta = find_pi_old_by_text_history_local(experiment_dir, cur_step, new_text)
            next_probs[new_eid] = float(p)
            reason = "revision_plan_no_sources_text_history"
            info.update(meta)

        else:
            if new_eid in pi_new_cur:
                next_probs[new_eid] = float(pi_new_cur[new_eid])
                reason = "same_id_inherit_pi_new"
                info.update({"from_id": new_eid, "from_step": cur_step})
            else:
                p, meta = find_pi_old_by_text_history_local(experiment_dir, cur_step, new_text)
                next_probs[new_eid] = float(p)
                reason = "no_revision_plan_entry_text_history"
                info.update(meta)

        debug["inheritance_next_step"][new_eid] = {
            "pi_old_next": float(next_probs[new_eid]),
            "reason": reason,
            **info,
        }

    return next_probs, debug