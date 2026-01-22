import json
import copy
import os
import re 

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from training_free_grpo.llm import LLM
from training_free_grpo.code.prompts import (
    SINGLE_QUERY_CRITIQUE_TEMPLATE, 
    SINGLE_ROLLOUT_SUMMARY_TEMPLATE,
    BATCH_EXPERIENCE_UPDATE_TEMPLATE
)


class ExperienceUpdater:
    def __init__(self):
        self.llm = LLM()

    def run(self, rollouts, experiences, save_dir, max_workers=16, given_ground_truth=True, only_partial_correct=True):
        # 1. Summarize trajectory for each rollout
        problem_to_summarized_rollouts = self._single_rollout_summary(
            rollouts=rollouts, 
            save_dir=save_dir, 
            max_workers=max_workers,
            given_ground_truth=given_ground_truth,
            only_partial_correct=only_partial_correct
        )

        # 2. Generate critique for each query
        critiques = self._single_query_critique(
            problem_to_summarized_rollouts=problem_to_summarized_rollouts, 
            experiences=experiences,
            save_dir=save_dir, 
            max_workers=max_workers,
            given_ground_truth=given_ground_truth,
            only_partial_correct=only_partial_correct
        )

        # 3. batch update experiences
        new_experiences = self._batch_update(
            experiences=experiences, 
            critiques=critiques, 
            save_dir=save_dir
        )

        # 4. assign new experience IDs
        new_experiences = {
            f"G{i}": exp for i, exp in enumerate(new_experiences.values())
        }
        return new_experiences


    def _single_rollout_summary(
        self,
        rollouts, 
        save_dir, 
        max_workers,
        given_ground_truth=True,
        only_partial_correct=True
    ):
        # check file existence
        filename = os.path.join(save_dir, "single_rollout_summary.json")
        if os.path.exists(filename):
            with open(filename) as f:
                results = json.load(f)
                if len(results) > 0:
                    print("Single rollout summary")
                    print("- File exists, loaded from:", filename)
                    return results

        # group by problems
        problems_to_rollouts = defaultdict(list)
        for each in rollouts:
            if "trajectories" in each and len(each["trajectories"]) > 0:
                problems_to_rollouts[each["problem"]].append(each)
        results = defaultdict(list)

        all_rollouts_to_process = []
        for rollouts in problems_to_rollouts.values():
            # all_rollouts_to_process.extend(rollouts)
            if given_ground_truth and only_partial_correct:
                # only for those partially correct
                scores = [each["reward"] for each in rollouts]
                avg_score = sum(scores) / len(scores)
                if avg_score > 0 and avg_score < 1:
                    all_rollouts_to_process.extend(rollouts)
            else:
                all_rollouts_to_process.extend(rollouts)

        def process(cur):
            try:
                execution_feedback = cur.get("execution_feedback", "")

                response = self.llm.chat(
                    SINGLE_ROLLOUT_SUMMARY_TEMPLATE.format(
                        trajectory=cur["trajectories"][0]["trajectory"], 
                        grade="This trajectory delivers **" + ("correct" if cur["reward"] == 1.0 else "wrong") + "** answer" + "\n" + execution_feedback, 
                    ) if given_ground_truth else
                    SINGLE_ROLLOUT_SUMMARY_TEMPLATE.format(
                        trajectory=cur["trajectories"][0]["trajectory"]
                    )
                )
                return {"trajectory_summary": response, **cur}
            except Exception as e:
                print(f"Warning: failed in single query critique, {e}")
                return None

        # parallel running
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_rollout = {executor.submit(process, cur): cur for cur in all_rollouts_to_process}
            for future in tqdm(
                as_completed(future_to_rollout), total=len(all_rollouts_to_process), desc="Single rollout summary"
            ):
                result = future.result()
                if result is not None:
                    problem = result["problem"]
                    results[problem].append(result)

        # write to file
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        return results


    def _single_query_critique(
        self,
        problem_to_summarized_rollouts, 
        experiences, 
        save_dir, 
        max_workers, 
        max_operations=1,
        given_ground_truth=True,
        only_partial_correct=True
    ):
        # check file existence
        filename = os.path.join(save_dir, "single_query_critique.json")
        if os.path.exists(filename):
            with open(filename) as f:
                results = json.load(f)
                if len(results) > 0:
                    print("Single query critique")
                    print("- File exists, loaded from:", filename)
                    return results

        all_rollouts = []
        for rollouts in problem_to_summarized_rollouts.values():
            if given_ground_truth and only_partial_correct:
                # only for those partially correct
                scores = [each["reward"] for each in rollouts]
                avg_score = sum(scores) / len(scores)
                if avg_score > 0 and avg_score < 1:
                    all_rollouts.append(rollouts)
            else:
                all_rollouts.append(rollouts)

        def process(rollouts_per_problem):
            try:
                problem = rollouts_per_problem[0]["problem"]
                formatted_trajectories = "\n\n".join([
                    f"Trajectory {i+1} (Answer {'correct' if each['reward'] else 'wrong'}):\n{each['trajectory_summary']}"
                    for i, each in enumerate(rollouts_per_problem)
                ])
                formatted_experiences = "\n".join([ f"[{i}]. {e}" for i, e in experiences.items() ]) if experiences else "None"
                
                response = self.llm.chat(
                    SINGLE_QUERY_CRITIQUE_TEMPLATE.format(
                        max_operations=max_operations,
                        problem=problem,
                        trajectories=formatted_trajectories,
                        experiences=formatted_experiences,
                    ) if given_ground_truth else
                    SINGLE_QUERY_CRITIQUE_TEMPLATE.format(
                        max_operations=max_operations,
                        problem=problem,
                        trajectories="\n\n".join([
                            f"Trajectory {i+1}:\n{each['trajectory_summary']}" for i, each in enumerate(rollouts_per_problem)
                        ]),
                        experiences=formatted_experiences
                    )
                )

                json_str = response
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0]
                elif "```" in response:
                    json_str = response.split("```")[1].split("```")[0]
                
                json_str = json_str.strip()
                # Invalid escape fix (\frac -> \\frac)
                json_str = re.sub(r'\\(?![/u"bfnrt\\])', r'\\\\', json_str)

                try:
                    operations = json.loads(json_str)
                except json.JSONDecodeError:
                    # Retry by finding the outermost brackets
                    match = re.search(r'(\{.*\}|\[.*\])', json_str, re.DOTALL)
                    if match:
                        extracted = match.group(0)
                        extracted = re.sub(r'\\(?![/u"bfnrt\\])', r'\\\\', extracted)
                        operations = json.loads(extracted)
                    else:
                        raise
                
                # Ensure operations is a list
                if isinstance(operations, dict):
                    operations = [operations]

                return {"rollouts": rollouts_per_problem, "critique": response, "operations": operations[:max_operations]}
            except Exception as e:
                print(f"Warning: failed in single query critique, {e}")
                return None

        # parallel running
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_case = {
                executor.submit(process, rollouts_per_problem): rollouts_per_problem
                for rollouts_per_problem in all_rollouts
            }
            for future in tqdm(as_completed(future_to_case), total=len(all_rollouts), desc="Single query critique"):
                result = future.result()
                if result is not None:
                    results.append(result)

        # write results
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        return results


    def _batch_update(
        self,
        experiences, 
        critiques, 
        save_dir,
        max_retries=3
    ):
        print("Batch update")
        filename = os.path.join(save_dir, "batch_update.json")
        if os.path.exists(filename):
            results = json.load(open(filename))
            print("- File exists, loaded from:", filename)
            return results
        
        # collect operations
        all_operations = []
        for each in critiques:
            all_operations.extend(each["operations"])
        print("- Num of operations to process:", len(all_operations))

        # split experiences
        candidate_experiences = copy.deepcopy(experiences)
        to_modify = []
        max_ID = 0
        
        for operation in all_operations:
            if not isinstance(operation, dict):
                continue
            if "option" not in operation:
                # print(f"Skipping invalid operation (no 'option' key): {operation}")
                continue

            if operation["option"] == "modify":
                if operation.get("modified_from") in candidate_experiences:
                    to_modify.append(operation)
            elif operation["option"] == "add":
                candidate_experiences[f"C{max_ID}"] = operation.get("experience", "")
                max_ID += 1

        print("- Num of added experiences:", max_ID)
        print("- Num of experiences to be modified:", len(to_modify))
        print("- Num of candidate experiences:", len(candidate_experiences))

        # use LLM to get the revision plan
        revision_plan = []
        response = "" 
        for _ in range(max_retries):
            try:
                response = self.llm.chat(
                    BATCH_EXPERIENCE_UPDATE_TEMPLATE.format(
                        experiences=candidate_experiences, 
                        updates=to_modify
                    )
                )
                
                # Robust parsing for revision plan as well
                json_str = response
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0]
                elif "```" in response:
                    json_str = response.split("```")[1].split("```")[0]
                json_str = json_str.strip()
                json_str = re.sub(r'\\(?![/u"bfnrt\\])', r'\\\\', json_str)
                
                revision_plan = json.loads(json_str)
                break
            except Exception:
                print("Warning: failed to decode in updating general experiences")

        # modify candidate experiences
        new_experiences = copy.deepcopy(candidate_experiences)
        for operation in revision_plan:
            try:
                opt = operation.get("option")
                
                if opt == "modify":
                    mod_from = operation.get("modified_from")
                    if mod_from:
                        new_experiences[mod_from] = operation.get("experience", "")
                
                elif opt == "merge":
                    merged_from = operation.get("merged_from", [])
                    for ID in merged_from:
                        if ID not in new_experiences:
                            continue
                    for ID in merged_from:
                        if ID in new_experiences:
                            del new_experiences[ID]
                    new_experiences[f"C{max_ID}"] = operation.get("experience", "")
                    max_ID += 1
            except Exception as e:
                print("Error: failed to complete experience update:", operation, "|", e)
        print("- Num of revised candidate experiences:", len(new_experiences))

        # write to file
        with open(filename, "w") as f:
            json.dump(
                {
                    "operations": all_operations,
                    "response": response,
                    "revision_plan": revision_plan,
                    "new_experiences": new_experiences,
                },
                f,
                indent=2,
            )
        return new_experiences