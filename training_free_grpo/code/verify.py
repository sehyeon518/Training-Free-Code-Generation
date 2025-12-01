import io
import ast
from contextlib import redirect_stdout
import traceback

def verify_func(sample: dict, tests: dict, timeout_sec=2.0) -> float:
    code = extract_code_from_response(sample["response"])
    if not code:
        reward = {
            "reward": 0.0,
            "passed": 0,
            "total": 0,
            "errors": ["no code extracted (invalid format)"],
            "penalty": 1.0
        }
        return reward["reward"]

    test_type = tests.get("type")
    if test_type == "check":
        if "def check(" in sample["response"]:
            print("Warning: 'def check' found in response, please ensure only solution code is provided.")
            reward = {"reward": 0.0, "passed": 0, "total": 0, "errors": ["The function 'check' should not be defined in the response."], "penalty": 1.0}
        if sample["reference_solutions"].count("def(") > 1:
            print("Warning: Multiple function definitions found in reference solutions.")
            reward =  {"reward": 0.0, "passed": 0, "total": 0, "errors": ["Multiple function definitions found in reference solutions."], "penalty": 1.0}
        reward = run_check_function(tests["code"], sample["response"])
    elif test_type == "stdin_stdout":
        reward = run_stdio(tests, sample["response"])

    return reward["reward"]


def extract_code_from_response(response: str) -> str:
    import re
    pattern = r"```python(.*?)```"
    m = re.search(pattern, response, re.S)
    if m:
        return m.group(1).strip()
    return ""
    

def run_check_function(test_code: str, code: str):
    import re
    code = "\n".join(code.splitlines()[1:-1])
    code = re.sub(r'\bList\s*\[[^\]]+\]\s*\(', 'list(', code)
    code = re.sub(r'\bList\s*\(', 'list(', code)
    ns = {}
    try:
        exec(code, ns)
    except Exception as e:
        return {"reward": 0.0, "passed": 0, "total": 0, "errors": f"response_exec_error: {e}"}

    candidates = [
        (name, obj) for name, obj in ns.items() if callable(obj) and not name.startswith("__") and hasattr(obj, "__code__")
    ]
    candidate_func = candidates[0][1]

    try:
        exec(test_code, ns)
    except Exception as e:
        return {"reward": 0.0, "passed": 0, "total": 0, "errors": f"testcode_exec_error: {e}"}
    
    total = 0
    passed = 0

    try:
        check_fn = ns["check"]
        total = test_code.count("assert")

        import inspect
        if len(inspect.signature(check_fn).parameters) == 1:
            check_fn(candidate_func)
        else:
            check_fn()
        passed = total
    except AssertionError as e:
        error = []
        passed = 0
        for line in test_code.splitlines():
            if "assert" in line:
                try:
                    exec(line.strip(), ns)
                    passed += 1
                except Exception:
                    error.append(f"Failed assertion: {line.strip()}")
                    pass
    except Exception as e:
        error = f"test_runtime_error: {e}"
        total = test_code.count("assert")
        passed = 0

    reward = 1 if passed == total else 0
    return {"reward": reward, "passed": passed, "total": total, "errors": error if passed != total else None}
    


def parse_input_to_args(arg_str: str):
    """Parse '(1,2)', '[1,2]', '3' safely into Python args.
    Returns a tuple of positional arguments.
    """
    arg_str = arg_str.strip()
    try:
        node = ast.literal_eval(arg_str)
    except Exception:
        return (arg_str,)  # treat raw string as single argument
    if isinstance(node, tuple):
        return node
    return (node,)


def run_stdio(tests: dict, code: str):
    code = "\n".join(code.splitlines()[1:-1])

    import io, sys

    ns = {}
    try:
        exec(code, ns)
    except Exception:
        return {"reward": 0.0, "passed": 0, "total": len(tests["input"]), "errors": "response_exec_error"}

    total = len(tests["input"])
    passed = 0
    errors = []

    for idx, (inp, expected_out) in enumerate(zip(tests["input"], tests["output"])):
        input_lines = inp.splitlines(keepends=True)
        input_iter = iter(input_lines)
        def fake_input(prompt=None):
            try:
                return next(input_iter).rstrip('\n')
            except StopIteration:
                return ""
        
        ns_local = ns.copy()
        ns_local["input"] = fake_input
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                funcs = [obj for name, obj in ns.items() if callable(obj) and not name.startswith("__")]
                if not funcs:
                    return {
                        "reward": 0.0,
                        "passed": 0,
                        "total": total,
                        "errors": "no_callable_function_found",
                    }

                target_func = funcs[0]
                target_func()
        except Exception as e:
            errors.append(f"test_{idx}_runtime_error: {e}")
            continue

        actual_out = buf.getvalue()
        if actual_out == expected_out:
            passed += 1
        else:
            errors.append(
                f"test_{idx}_failed: expected={expected_out!r}, got={actual_out!r}"
            )

    reward = 1 if passed == total else 0
    return {
        "reward": reward, "passed": passed, "total": total, "errors": errors if passed != total else None,}
    

if __name__ == "__main__":
    from dataset import load_data
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="[CodeContests, HumanEval, HumanEvalPlus, MBPP, MBPPPlus]")
    args = parser.parse_args()
    data = load_data(args.dataset)

    for i in range(len(data)):
        sample = data[i]
        sample["response"] = "```python\n" + data[i]["reference_solutions"][0] + "\n```"
        sample["reward"] = verify_func(sample, sample["tests"])
        if sample["reward"] < 1.0:
            print(f"Sample {i} failed verification: {sample['reward']}")
        