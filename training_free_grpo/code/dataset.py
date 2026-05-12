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
                "problem": each["prompt"].lstrip("\n"),
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
            defs = set(re.findall(r'^\s*def\s+([A-Za-z_]\w*)\s*\(', each["code"], re.MULTILINE))
            called = set(re.findall(r'([A-Za-z_]\w*)\s*\(', " ".join(each["test_list"])))

            function_name = defs & called
            if len(function_name) != 1:
                continue
            function_name = function_name.pop()
            if function_name in ["candidate", "check"]:
                continue
            test_list = list(each["test_list"])

            pattern = rf'\b{re.escape(function_name)}\s*\('

            for i in range(len(test_list)):
                test_list[i] = re.sub(pattern, "candidate(", test_list[i])
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
            defs = set(re.findall(r'^\s*def\s+([A-Za-z_]\w*)\s*\(', each["code"], re.MULTILINE))
            called = set(re.findall(r'([A-Za-z_]\w*)\s*\(', " ".join(each["test_list"])))

            function_name = defs & called
            if len(function_name) != 1:
                continue
            function_name = function_name.pop()
            if function_name in ["candidate", "check"]:
                continue

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

    elif name == "livecodebench":
        dataset = load_dataset("livecodebench/code_generation")
        data = []
        duplicated_check = set()
        for each in dataset["test"]:
            # Skip duplicates
            question = each.get("question_content", "")
            marker = "Sample Input 1"
            if marker in question:
                question = question[:question.index(marker)].rstrip()

            if question in duplicated_check:
                continue
            duplicated_check.add(question)

            # Extract public test cases
            public_test_cases = each.get("public_test_cases", [])
            public_test_cases = json.loads(public_test_cases) 

            inputs = [test.get("input", "") for test in public_test_cases]
            outputs = [test.get("output", "") for test in public_test_cases]
            
            # Extract reference solutions
            reference_solutions = each.get("solutions", [each.get("solution", "")])
            if isinstance(reference_solutions, str):
                reference_solutions = [reference_solutions]
            
            # Filter out empty solutions
            reference_solutions = [sol for sol in reference_solutions if sol]
            # if not reference_solutions:
            #     continue

            d = {
                "problem": question,
                "reference_solutions": reference_solutions,
                "tests": {
                    "type": "stdin_stdout",
                    "input": inputs,
                    "output": outputs,
                },
                "language": each.get("language", "python").lower(),
            }
            data.append(d)

        return data
    
    raise ValueError(f"Unsupported dataset: {name}. Supported datasets are: CodeContests, HumanEval, HumanEvalPlus, MBPP, MBPPPlus, livecodebench.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="[CodeContests, HumanEval, HumanEvalPlus, MBPP, MBPPPlus, livecodebench]")
    args = parser.parse_args()
    data = load_data(args.dataset)
    print(f"Loaded {len(data)} samples from {args.dataset} dataset.")
    print(data[0]["problem"])
    print(data[0]["reference_solutions"] if isinstance(data[0]["reference_solutions"], str) else data[0]["reference_solutions"][0])
    print(data[0]["tests"])
