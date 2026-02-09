# 结构化评估系统 (Structured Evaluation System)

这是一个模块化的模型评估系统，将数据集处理和评估逻辑分离到独立的处理器中。

## 文件结构

```
RL/
├── datasets/                    # 数据集处理器模块
│   ├── __init__.py             # 导入所有处理器和注册表
│   ├── base.py                 # 基础数据集处理器类
│   ├── judge.py                # 共享的GPT Judge评估模块
│   ├── mmlu.py                 # MMLU多选题数据集处理器
│   ├── gsm8k.py                # GSM8K数学问题数据集处理器
│   ├── strongreject.py         # StrongReject安全数据集处理器
│   ├── harmbench.py            # HarmBench安全数据集处理器
│   ├── pku_safe.py             # PKU-SafeRLHF安全数据集处理器
│   ├── truthfulqa.py           # TruthfulQA真实性评估数据集处理器
│   ├── custom.py               # 自定义漏洞报告数据集处理器
│   └── README.md               # 本文档
├── evaluation_refactored.py    # 重构的主评估器
├── evaluation_example.py       # 使用示例
└── evaluation.py               # 原始评估器（保留）
```

## 主要特性

### 1. 模块化设计
- 每个数据集都有独立的处理器
- 统一的基础类接口
- 共享的Judge评估模块

### 2. 灵活的数据集选择
```python
# 只评估学术能力
academic_datasets = ["mmlu", "gsm8k"]

# 只评估安全性和真实性
safety_datasets = ["strongreject", "harmbench", "pku_safe", "truthfulqa", "olmoe_fwo"]

# 全面评估
all_datasets = ["mmlu", "gsm8k", "strongreject", "harmbench", "pku_safe", "truthfulqa", "olmoe_fwo", "custom"]
```

### 3. 统一的评估接口
```python
evaluator = ModelEvaluator(
    model_name="your-model-path",
    output_dir="./results",
    prefix="MODEL_NAME"
)

scores = evaluator.run_evaluation(
    dataset_names=["mmlu", "gsm8k"],
    num_samples_to_eval=100,
    batch_size=16
)
```

### 4. OLMOE-FWO 数据集评估
```python
evaluator = ModelEvaluator(
    model_name="your-model-path",
    output_dir="./results",
    prefix="MODEL_NAME",
    olmoe_fwo_path="/path/to/OLMOE-FWO.json"  # 指定数据集路径
)

scores = evaluator.run_evaluation(
    dataset_names=["olmoe_fwo"],
    num_samples_to_eval=50,
    batch_size=8,
    max_workers=3  # 并发 judge 评估
)
```

## 数据集处理器

### MMLU (mmlu.py)
- **类型**: 多选题 (Multiple Choice)
- **评估**: 提取选择答案 (A/B/C/D)
- **格式**: `\\boxed{A}` 

### GSM8K (gsm8k.py)  
- **类型**: 数学问题 (Math Problem)
- **评估**: 数值比较
- **格式**: `\\boxed{123}` vs `#### 123`

### 安全数据集 (strongreject.py, harmbench.py, pku_safe.py)
- **类型**: 安全评估 (Safety)
- **评估**: GPT Judge 评分 (0-5分)
- **共享**: judge.py 模块

### TruthfulQA (truthfulqa.py)
- **类型**: 真实性评估 (Truthfulness)
- **评估**: GPT Judge 真实性评分 (0-5分)
- **特点**: 检测常见误解和事实错误

### OLMOE-FWO (olmoe_fwo.py)
- **类型**: 越狱攻击安全评估 (Jailbreak Attack Safety)
- **来源**: OLMOE-FWO.json 数据集文件
- **评估**: GPT Judge 评分 (0-5分)，使用与 HarmBench 相同的 judge
- **特点**: 使用完整的越狱攻击 prompt (all_prompt)，但用原始 goal 进行评估
- **指标**: Safety Rate (安全率 = score=0 的样本数 / 总样本数)

### 自定义数据集 (custom.py)
- **类型**: 安全评估 (Safety)  
- **来源**: 漏洞报告JSON文件
- **评估**: GPT Judge 评分

## 使用方法

### 基本使用
```python
from evaluation_refactored import ModelEvaluator

evaluator = ModelEvaluator(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    output_dir="./results",
    prefix="TEST"
)

scores = evaluator.run_evaluation(
    dataset_names=["mmlu", "gsm8k"],
    num_samples_to_eval=50
)
```

### 高级配置
```python
evaluator = ModelEvaluator(
    model_name="/path/to/fine-tuned/model",
    output_dir="/path/to/results",
    temperature=0.7,
    top_p=0.9,
    vllm=True,  # 使用vLLM加速
    prefix="MY_MODEL",
    vulnerability_report_path="./vulnerability_report.json",
    olmoe_fwo_path="/path/to/OLMOE-FWO.json"  # OLMOE-FWO数据集路径
)

scores = evaluator.run_evaluation(
    dataset_names=["mmlu", "gsm8k", "strongreject", "truthfulqa", "olmoe_fwo", "custom"],
    num_samples_to_eval=200,
    batch_size=32,
    max_workers=8
)
```

## 添加新数据集

要添加新的数据集处理器：

1. 创建新的处理器文件 `datasets/your_dataset.py`
2. 继承 `BaseDatasetHandler` 类
3. 实现必需的方法：
   - `get_dataset_name()`
   - `load_dataset()`
   - `prepare_prompt()`
   - `evaluate_response()`
4. 在 `datasets/__init__.py` 中注册新处理器

### 示例：
```python
from .base import BaseDatasetHandler

class YourDatasetHandler(BaseDatasetHandler):
    def get_dataset_name(self) -> str:
        return "your_dataset"
    
    def load_dataset(self) -> Dataset:
        # 加载数据集逻辑
        pass
    
    def prepare_prompt(self, sample: Dict[str, Any]) -> str:
        # 准备提示词
        pass
    
    def evaluate_response(self, prompt: str, response: str, sample: Dict[str, Any]) -> Dict[str, Any]:
        # 评估响应
        pass
```

## 输出文件

每个数据集的评估结果都会保存为独立的JSON文件：
- `evaluation_results-mmlu(PREFIX).json`
- `evaluation_results-gsm8k(PREFIX).json`
- `evaluation_results-strongreject(PREFIX).json`
- 等等

结果包含详细的样本级评估信息和总体统计。
