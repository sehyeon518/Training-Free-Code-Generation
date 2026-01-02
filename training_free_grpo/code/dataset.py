import os
import re
import json
import random
from typing import List, Dict, Any
from datasets import load_dataset


def load_data(name: str) -> List[Dict[str, Any]]:

    if name == "CodeContests":    
        dataset = load_dataset("deepmind/code_contests", split="test")
        data = []
        for each in dataset:
            d = {
                "problem": each["description"],
                "reference_solutions": each["solutions"]["solution"],  # <class 'list'>
                "tests": {
                    "type": "stdin_stdout",
                    "input": each["public_tests"]["input"] + each["private_tests"]["input"],
                    "output": each["public_tests"]["output"] + each["private_tests"]["output"],
                },
                "language": "cpp", 
            }
            data.append(d)
        return data
    
    elif name == "HumanEval":
        dataset = load_dataset("openai/openai_humaneval", split="test")
        data = []
        for each in dataset:
            if each["prompt"].count("def ") > 1:
                continue
            lines = each["prompt"].splitlines()
            def_indices = [i for i, line in enumerate(lines) if line.strip().startswith("def ")]
            reference_solution = []
            reference_solution.extend(lines[:def_indices[0] + 1])
            reference_solution = "\n".join(reference_solution) + "\n" + each["canonical_solution"]

            d = {
                "problem": each["prompt"],
                "reference_solutions": [reference_solution],
                "tests": {
                    "type": "check",  # assert in check function
                    "code": "def check" + each["test"].split("def check")[1],
                },
                "language": "python",
            }
            data.append(d)
        return data
    
    elif name == "HumanEvalPlus":
        dataset = load_dataset("evalplus/humanevalplus", split="test")
        data = []
        for each in dataset:
            if each["prompt"].count("def ") > 1:
                continue
            lines = each["prompt"].splitlines()
            def_indices = [i for i, line in enumerate(lines) if line.strip().startswith("def ")]
            reference_solution = []
            reference_solution.extend(lines[:def_indices[0] + 1])
            reference_solution = "\n".join(reference_solution) + "\n" + each["canonical_solution"]

            d = {
                "problem": each["prompt"],
                "reference_solutions": [reference_solution],
                "tests": {
                    "type": "check",  # assert in check function
                    "code": each["test"],
                },
                "language": "python",
            }
            data.append(d)
        data = sorted(data, key=lambda x: len(x["problem"]))
        return data

    elif name == "MBPP":
        dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        data = []
        duplicated_check = set()
        for each in dataset:
            m = re.search(r'^\s*def\s+([A-Za-z_]\w*)\s*\(', each["code"], re.MULTILINE)
            function_name = m.group(1) if m else None
            test_list = list(each["test_list"])
            if function_name:
                for i in range(len(test_list)):
                    test_list[i] = test_list[i].replace(f"{function_name}(", "candidate(")
            code = """def check(candidate):\n    """ + "\n    ".join(test_list)

            if each["prompt"] in duplicated_check:
                continue
            duplicated_check.add(each["prompt"])

            tests = {
                "type": "check",
                "code": code,
            }
            if each.get("test_imports"):
                tests["code"] = each["test_imports"][0] + "\n" + tests["code"]

            d = {
                "problem": each["prompt"],
                "reference_solutions": [each["code"]],
                "tests": tests,
                "language": "python",
            }
            data.append(d)
        return data
    
    elif name == "MBPPPlus":
        dataset = load_dataset("evalplus/mbppplus")
        data = []
        duplicated_check = set()
        for each in dataset["test"]:
            m = re.search(r'^\s*def\s+([A-Za-z_]\w*)\s*\(', each["code"], re.MULTILINE)
            function_name = m.group(1) if m else None
            test_list = list(each["test_list"])
            if function_name:
                for i in range(len(test_list)):
                    test_list[i] = test_list[i].replace(f"{function_name}(", "candidate(")
            code = """def check(candidate):\n    """ + "\n    ".join(test_list)

            if each["prompt"] in duplicated_check:
                continue
            duplicated_check.add(each["prompt"])
            d = {
                "problem": each["prompt"],
                "reference_solutions": [each["code"]],
                "tests": {
                    "type": "check",
                    "code": code,
                },
                "language": "python",
            }
            data.append(d)
        return data
    
    raise ValueError(f"Unsupported dataset: {name}. Supported datasets are: CodeContests, HumanEval, HumanEvalPlus, MBPP, MBPPPlus.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="[CodeContests, HumanEval, HumanEvalPlus, MBPP, MBPPPlus]")
    args = parser.parse_args()
    data = load_data(args.dataset)
    print(f"Loaded {len(data)} samples from {args.dataset} dataset.")
    print(data[0]["problem"])
    print(data[0]["reference_solutions"] if isinstance(data[0]["reference_solutions"], str) else data[0]["reference_solutions"][0])
    print(data[0]["tests"])
