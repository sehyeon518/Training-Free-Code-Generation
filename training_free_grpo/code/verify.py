import io
import ast
import json
import os
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
import traceback


def verify_func(sample: dict, tests: dict, timeout_sec: float = 2.0) -> dict:
    """
    return shape: {
        "reward": float,
        "passed": int,
        "total": int,
        "errors": list or None,
        "execution_events": list
        }
    """
    language = sample.get("language", "python")

    code = extract_code_from_response(sample["response"])
    if not code:
        return {
            "reward": 0.0,
            "passed": 0,
            "total": 0,
            "errors": ["no code extracted (invalid format)"],
            "execution_events": [],
        }

    sample["response"] = code
    test_type = tests.get("type")

    # The check type is for HumanEval/BMPP → Only supports Python
    if test_type == "check":
        if language != "python":
            return {
                "reward": 0.0,
                "passed": 0,
                "total": 0,
                "errors": [f"check-type tests only support python, got {language}"],
                "execution_events": [],
            }

        if "def check(" in sample["response"]:
            print("Warning: 'def check' found in response, please ensure only solution code is provided.")
            return {
                "reward": 0.0,
                "passed": 0,
                "total": 0,
                "errors": ["The function 'check' should not be defined in the response."],
                "execution_events": [],
            }
        
        if isinstance(sample.get("reference_solutions"), list) and any(
            ("def(" in rs) for rs in sample["reference_solutions"]
        ):
            print("Warning: Multiple function definitions found in reference solutions.")

        return run_check_function(tests["code"], sample["response"])

    elif test_type == "stdin_stdout":
        return run_stdio_python(tests, sample["response"], timeout_sec) 
    else:
        reward = {
            "reward": 0.0,
            "passed": 0,
            "total": 0,
            "errors": [f"unknown test type: {test_type}"],
            "penalty": 1.0,
        }

    return reward


def extract_code_from_response(response: str) -> str:
    """
    Extracts the first code block from a response string, which can be in any language format like:
    ```python
    ...
    ```
    ```cpp
    ...
    ```
    If no code block is found, the entire response is returned as is.
    """
    import re

    pattern = r"```(?:\w+)?\s*(.*?)```"
    m = re.search(pattern, response, re.S)
    if m:
        return m.group(1).strip()
    return response.strip()


def safe_repr(value, max_len=200):
    try:
        s = repr(value)
    except Exception:
        s = object.__repr__(value)
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def run_check_function(test_code: str, code: str) -> dict:
    import builtins
    
    ns = {}

    def _disabled_input(*args, **kwargs):
        raise RuntimeError("input() is disabled")

    builtins_backup = builtins.__dict__.copy()
    builtins.input = _disabled_input
    try:
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
            exec(code, ns)
    except Exception as e:
        return {"reward": 0.0, "passed": 0, "total": 0, "errors": [f"LLM response execution error: {e}"], "execution_events": [],}

    candidates = [
        (name, obj)
        for name, obj in ns.items()
        if callable(obj) and not name.startswith("__") and hasattr(obj, "__code__")
    ]
    if not candidates:
        return {"reward": 0.0, "passed": 0, "total": 0, "errors": ["no callable candidate found"], "execution_events": [],}
    
    candidate_func = candidates[0][1]

    try:
        exec(test_code, ns)
    except Exception as e:
        return {"reward": 0.0, "passed": 0, "total": 0, "errors": [f"test code execution error: {e}"], "execution_events": [],}
    
    total = 0
    passed = 0
    error = None

    execution_events = []

    def tracer(frame, event, arg):
        if frame.f_code is candidate_func.__code__:
            record = {
                "event": event,
                "lineno": frame.f_lineno,
                "globals": frame.f_globals.get("__name__", ""),
                "locals": {k: safe_repr(v) for k, v in frame.f_locals.items()},
            }
            if event == "return":
                record["return"] = safe_repr(arg)
            elif event == "exception":
                exc_type, exc_value, _ = arg
                record["exception_type"] = safe_repr(exc_type)
                record["exception_value"] = safe_repr(exc_value)

            if event == "line":
                return
            execution_events.append(record)
        return tracer
    
    try:
        check_fn = ns["check"]
        total = test_code.count("assert")

        import inspect
        sys.settrace(tracer)
        try:
            if len(inspect.signature(check_fn).parameters) == 1:
                check_fn(candidate_func)
            else:
                check_fn()
            passed = total
        except AssertionError: # 하나씩 실행하면서 어떤 테스트 케이스가 실패했는지 기록
            error = []
            passed = 0
            for line in test_code.splitlines():
                if "assert" in line:
                    try:
                        exec(line.strip(), ns)
                        passed += 1
                    except Exception:
                        error.append(f"Failed assertion: {line.strip()}")
                        execution_events = []
        except Exception as e:
            error = [f"test_runtime_error: {e}"]
            total = test_code.count("assert")
            passed = 0
            execution_events = []
        finally:
            sys.settrace(None)
    finally:
        if sys.gettrace() is tracer:
            sys.settrace(None)
        
    execution_events = execution_events[:5] if len(execution_events) > 5 else None
    reward = 1 if passed == total else passed / total
    result = {
        "reward": reward, "passed": passed, "total": total, "errors": error if passed != total else None, "execution_events": execution_events
    }

    return result


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


# ===== Python stdin/stdout 실행 (혹시 있을 Python용 stdin_stdout 데이터셋 대비) =====

def run_stdio_python(tests: dict, code: str, timeout_sec: float = 2.0) -> dict:
    import os
    import subprocess
    import tempfile

    total = len(tests["input"])
    if total == 0:
        return {
            "reward": 0.0,
            "passed": 0,
            "total": 0,
            "errors": ["no tests"],
            "execution_events": [],
        }

    passed = 0
    errors = []

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "main.py")

        with open(src_path, "w", encoding="utf-8") as f:
            f.write(code)

        for idx, (inp, expected_out) in enumerate(zip(tests["input"], tests["output"])):
            try:
                run_proc = subprocess.run(
                    [sys.executable, src_path],
                    input=inp,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout_sec,
                )
            except subprocess.TimeoutExpired:
                errors.append(f"test_{idx}_timeout")
                continue

            if run_proc.returncode != 0:
                errors.append(f"test_{idx}_runtime_error: {run_proc.stderr}")
                continue

            actual_out = run_proc.stdout

            if normalize_output(actual_out) == normalize_output(expected_out):
                passed += 1
            else:
                errors.append(
                    f"test_{idx}_failed: expected={expected_out!r}, got={actual_out!r}"
                )

    reward = 1.0 if passed == total else passed / total

    return {
        "reward": reward,
        "passed": passed,
        "total": total,
        "errors": errors if passed != total else None,
        "execution_events": [],
    }


def normalize_output(s: str) -> str:
    return "\n".join(line.rstrip() for line in s.strip().splitlines())


def run_stdio_cpp(tests: dict, code: str, timeout_sec: float = 2.0) -> dict:
    """
    C++ 코드를 g++로 컴파일해서, 각 테스트케이스의 input을 stdin으로 넣고
    stdout을 tests["output"]과 비교한다.
    """
    total = len(tests["input"])
    if total == 0:
        return {"reward": 0.0, "passed": 0, "total": 0, "errors": ["no tests"]}

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "main.cpp")
        exe_path = os.path.join(tmpdir, "main.out")

        with open(src_path, "w", encoding="utf-8") as f:
            f.write(code)

        try:
            compile_proc = subprocess.run(
                ["g++", "-std=c++17", "-O2", src_path, "-o", exe_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return {
                "reward": 0.0,
                "passed": 0,
                "total": total,
                "errors": ["compile_timeout"],
            }

        if compile_proc.returncode != 0:
            return {
                "reward": 0.0,
                "passed": 0,
                "total": total,
                "errors": [f"compile_error: {compile_proc.stderr}"],
            }

        # 각 테스트 실행
        passed = 0
        errors = []

        for idx, (inp, expected_out) in enumerate(zip(tests["input"], tests["output"])):
            try:
                run_proc = subprocess.run(
                    [exe_path],
                    input=inp,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout_sec,
                )
            except subprocess.TimeoutExpired:
                errors.append(f"test_{idx}_timeout")
                continue

            if run_proc.returncode != 0:
                errors.append(f"test_{idx}_runtime_error: {run_proc.stderr}")
                continue

            actual_out = run_proc.stdout
            if actual_out == expected_out:
                passed += 1
            else:
                errors.append(
                    f"test_{idx}_failed: expected={expected_out!r}, got={actual_out!r}"
                )

        reward = 1 if passed == total else 0
        return {
            "reward": reward,
            "passed": passed,
            "total": total,
            "errors": errors if passed != total else None,
        }


if __name__ == "__main__":
    from dataset import load_data
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="[CodeContests, HumanEval, HumanEvalPlus, MBPP, MBPPPlus]",
    )
    args = parser.parse_args()
    data = load_data(args.dataset)

    for i in range(1):
        sample = data[i]
        # sample["response"] = sample["reference_solutions"][0] # True case
        sample["response"] = "from typing import List\ndef add(numbers: List[float], threshold: float) -> bool:\n    summation = sum(numbers)\n    return summation" # False case
        sample["language"] = sample.get("language", "python")
        res = verify_func(sample, sample["tests"])
        print(f"Sample {i} verification: {res}")
