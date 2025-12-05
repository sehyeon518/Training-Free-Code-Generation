import io
import ast
import os
import subprocess
import tempfile
from contextlib import redirect_stdout
import traceback


def verify_func(sample: dict, tests: dict, timeout_sec: float = 2.0) -> dict:
    language = sample.get("language", "python")

    code = extract_code_from_response(sample["response"])
    if not code:
        return {
            "reward": 0.0,
            "passed": 0,
            "total": 0,
            "errors": ["no code extracted (invalid format)"],
            "penalty": 1.0,
        }

    sample["response"] = code
    test_type = tests.get("type")

    # check 타입은 HumanEval/MBPP용 → Python만 지원
    if test_type == "check":
        if language != "python":
            return {
                "reward": 0.0,
                "passed": 0,
                "total": 0,
                "errors": [f"check-type tests only support python, got {language}"],
                "penalty": 1.0,
            }

        if "def check(" in sample["response"]:
            print("Warning: 'def check' found in response, please ensure only solution code is provided.")
            return {
                "reward": 0.0,
                "passed": 0,
                "total": 0,
                "errors": ["The function 'check' should not be defined in the response."],
                "penalty": 1.0,
            }
        
        if isinstance(sample.get("reference_solutions"), list) and any(
            ("def(" in rs) for rs in sample["reference_solutions"]
        ):
            print("Warning: Multiple function definitions found in reference solutions.")

        reward = run_check_function(tests["code"], sample["response"])

    elif test_type == "stdin_stdout":
        if language in ("cpp", "c++"):
            reward = run_stdio_cpp(tests, sample["response"], timeout_sec=timeout_sec)
        else:
            # 혹시 python stdin_stdout 데이터셋이 있을 경우 대비
            reward = run_stdio_python(tests, sample["response"])
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
    ```python
    ...
    ```
    ```cpp
    ...
    ```
    처럼 어떤 언어든 첫 번째 코드블럭만 떼어오는 함수.
    코드블럭이 없으면 전체 응답을 그대로 반환.
    """
    import re

    pattern = r"```(?:\w+)?\s*(.*?)```"
    m = re.search(pattern, response, re.S)
    if m:
        return m.group(1).strip()
    return response.strip()


def run_check_function(test_code: str, code: str) -> dict:
    print(code)
    ns = {}
    try:
        exec(code, ns)
    except Exception as e:
        return {"reward": 0.0, "passed": 0, "total": 0, "errors": f"response_exec_error: {e}"}

    candidates = [
        (name, obj)
        for name, obj in ns.items()
        if callable(obj) and not name.startswith("__") and hasattr(obj, "__code__")
    ]
    if not candidates:
        return {
            "reward": 0.0,
            "passed": 0,
            "total": 0,
            "errors": "no_callable_candidate_found",
        }
    candidate_func = candidates[0][1]

    try:
        exec(test_code, ns)
    except Exception as e:
        return {"reward": 0.0, "passed": 0, "total": 0, "errors": f"testcode_exec_error: {e}"}
    
    total = 0
    passed = 0
    error = None

    try:
        check_fn = ns["check"]
        total = test_code.count("assert")

        import inspect
        if len(inspect.signature(check_fn).parameters) == 1:
            check_fn(candidate_func)
        else:
            check_fn()
        passed = total
    except AssertionError:
        error = []
        passed = 0
        for line in test_code.splitlines():
            if "assert" in line:
                try:
                    exec(line.strip(), ns)
                    passed += 1
                except Exception:
                    error.append(f"Failed assertion: {line.strip()}")
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


# ===== Python stdin/stdout 실행 (혹시 있을 Python용 stdin_stdout 데이터셋 대비) =====

def run_stdio_python(tests: dict, code: str) -> dict:
    import sys

    ns = {}
    try:
        exec(code, ns)
    except Exception as e:
        return {
            "reward": 0.0,
            "passed": 0,
            "total": len(tests["input"]),
            "errors": f"response_exec_error: {e}",
        }

    total = len(tests["input"])
    passed = 0
    errors = []

    for idx, (inp, expected_out) in enumerate(zip(tests["input"], tests["output"])):
        input_lines = inp.splitlines(keepends=True)
        input_iter = iter(input_lines)

        def fake_input(prompt=None):
            try:
                return next(input_iter).rstrip("\n")
            except StopIteration:
                return ""
        
        ns_local = ns.copy()
        ns_local["input"] = fake_input
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                funcs = [
                    obj for name, obj in ns.items()
                    if callable(obj) and not name.startswith("__")
                ]
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
        "reward": reward,
        "passed": passed,
        "total": total,
        "errors": errors if passed != total else None,
    }


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
        # 테스트용: 여기서는 python 예시를 넣었지만,
        # 실제 CodeContests 돌릴 때는 language="cpp" + C++ 코드가 들어온다고 가정.
        sample["response"] = "print('hello')"
        sample["language"] = sample.get("language", "python")
        res = verify_func(sample, sample["tests"])
        print(f"Sample {i} verification: {res}")
