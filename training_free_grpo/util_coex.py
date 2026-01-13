import os
import math
import json
import random

def sample_experiences(experiences, experience_probs, k, rng):
    if k <= 0 or not experiences:
        return {}

    ids = list(experiences.keys())
    probs = [experience_probs[i] for i in ids]

    # safety: normalize again (cheap, avoids silent bugs)
    total = sum(probs)
    if total <= 0:
        # fallback to uniform
        probs = [1.0 / len(probs)] * len(probs)
    else:
        probs = [p / total for p in probs]

    selected_ids = []

    # copy lists for without-replacement sampling
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


# def sample_experiences(experiences, k, rng):
#     if not experiences:
#         return {}
    
#     experience_items = list(experiences.items())

#     if len(experience_items) <= k:
#         return dict(experience_items)
    
#     sampled_items = rng.sample(experience_items, k)
#     return dict(sampled_items)


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
        with open(path, "r") as f:
            experience_probs = json.load(f)
    else:
        experience_probs = {}

    n = len(experience_probs)
    if n == 0:
        if len(experiences) > 0:
            uniform_p = 1.0 / len(experiences)
            experience_probs = {exp_id: uniform_p for exp_id in experiences}
            json.dump(experience_probs, open(path, "w"), indent=2)
            return experience_probs
        return {}
    
    uniform_p = 1.0 / n
    for exp_id in experiences.keys():
        if exp_id not in experience_probs:
            experience_probs[exp_id] = uniform_p
    
    experience_probs = {experience_id: experience_probs[experience_id] for experience_id in experiences}

    total = sum(experience_probs.values())

    for exp_id in experience_probs:
        experience_probs[exp_id] /= total

    return experience_probs


def inherit_experience_probs(
        old_experiences, old_probs, batch_update_result,
        operations,
        alpha_add=0.5, delta_modify=0.1, alpha_merge=0.6,
        min_prob=1e-6,
    ):
    # Seohee TODO: implement proper inheritance logic
    # Merge: avg
    # Modify: inherit

    new_experience_probs = {
        f"G{i}": 0.5 for i in range(len(batch_update_result["new_experiences"].values()))
    }
    return new_experience_probs

            


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


def compute_next_experience_probs(
        old_experiences, old_probs, batch_update_result,
        operations,
        rollout_scores,
        alpha_add=0.5, delta_modify=0.1, alpha_merge=0.6,
        eta=10.0, min_prob=13-6,
    ):

    inherited_probs = inherit_experience_probs(
        old_experiences, old_probs, batch_update_result, operations,
        alpha_add, delta_modify, alpha_merge,
        min_prob
    )

    new_probs = update_experience_probs(
        inherited_probs,
        rollout_scores,
        eta, min_prob
    )
    # TODO: Rewrite inheritance logic (Seohee)
    # Revise arguments, logic, etc.

    return new_probs