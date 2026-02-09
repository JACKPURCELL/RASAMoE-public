import json

with open("/home/stufs1/jiachliang/FlipAttack/reproduce_result/FlipAttack-FWO-CoT-LangGPT-Few-shot-Qwen3-30B-A3B-advbench-0_519-final.json", "r") as f:
    data = json.load(f)

count = 0
test_data = []
for item in data:
    if item["judge_success_gpt4"] != 0:
        count += 1
        if count > 128:
            test_data.append(item)
        if len(test_data) == 100:
            break
with open("qwen_fwo_test_data.json", "w") as f:
    json.dump(test_data, f)
print(count)
print(len(test_data))