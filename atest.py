from datasets import load_dataset

# Load dataset
dataset = load_dataset("deepmind/code_contests", split="test")

# Show a few samples (예: 처음 2개만 보기)
num_samples = 2

for i, each in enumerate(dataset):
    if i >= num_samples:
        break

    print("="*80)
    print(f"[Sample {i}]")

    # 문제 설명
    print("\n--- Problem Description ---")
    print(each["description"])

    # 레퍼런스 솔루션 (list 형태)
    print("\n--- Reference Solutions (List of solutions) ---")
    for idx, sol in enumerate(each["solutions"]["solution"]):
        print(f"Solution {idx}:\n{sol}\n")

    # 테스트 정보
    print("\n--- Tests (stdin/stdout) ---")
    public_in  = each["public_tests"]["input"]
    public_out = each["public_tests"]["output"]
    private_in = each["private_tests"]["input"]
    private_out = each["private_tests"]["output"]

    print("Public Testcases:")
    for inp, out in zip(public_in, public_out):
        print("Input:")
        print(inp)
        print("Output:")
        print(out)
        print("-"*40)

    print("Private Testcases:")
    for inp, out in zip(private_in, private_out):
        print("Input:")
        print(inp)
        print("Output:")
        print(out)
        print("-"*40)

    print("="*80)
