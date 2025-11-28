import os
import json
import random
from typing import List, Dict, Any
from datasets import load_dataset


def load_data(name: str) -> List[Dict[str, Any]]:

    if name == "CodeContests":    
        dataset = load_dataset("deepmind/code_contests", split="test")
        data = []
        for each in dataset:
            d = {"problem": each["description"],
                 "reference_solutions": each["solutions"]["solution"], # <class 'list'>
                 "tests": []}
            d["tests"] = {
                "type": "stdin_stdout",
                "input": each["public_tests"]["input"] + each["private_tests"]["input"],
                "output": each["public_tests"]["output"] + each["private_tests"]["output"]
            }

            data.append(d)
            
        return data
    
    elif name == "HumanEval":
        dataset = load_dataset("openai/openai_humaneval", split="test")
        data = []
        for each in dataset:
            d = {"problem": each["prompt"],
                 "reference_solutions": [each["canonical_solution"]],
                 "tests": []}
            d["tests"] = {
                "type": "check", # assert in check function
                "code": "def check" + each["test"].split("def check")[1]
            }

            data.append(d)

        return data
    
    elif name == "HumanEvalPlus":
        dataset = load_dataset("evalplus/humanevalplus", split="test")
        data = []
        for each in dataset:
            d = {"problem": each["prompt"],
                 "reference_solutions": each["canonical_solution"],  # <class 'list'>
                 "tests": []}
            d["tests"] = {
                "type": "check",
                "code": each["test"]
            }

            data.append(d)

        return data

    elif name == "MBPP":
        dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        data = []
        for each in dataset:
            d = {"problem": each["prompt"],
                 "reference_solutions": [each["code"]],
                 "tests": []}
            code = """def check():\n    """ + "\n    ".join(each["test_list"])
            d["tests"] = {
                "type": "check",
                "code": code
            }

            data.append(d)

        return data
    
    elif name == "MBPPPlus":
        dataset = load_dataset("evalplus/mbppplus")
        data = []
        for each in dataset["test"]:
            d = {"problem": each["prompt"],
                 "reference_solutions": each["code"],  # <class 'list'>
                 "tests": []}
            code = "def check():\n" + "\n".join("    " + line for line in each["test"].strip().splitlines())
            d["tests"] = {
                "type": "check",
                "code": code
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