# BabyConceptLM 论文分析发布包

[English README](README.md)

本目录是可独立整理为 GitHub 仓库的论文分析代码包。它对应 `docs/provenance.json` 

## 最快运行方式

建议使用 Python 3.11，在本目录运行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tested.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
babyconceptlm-analysis reproduce --reference-dir examples/reference --output-dir outputs/reference
```

安装依赖需要取得相应软件包；最后一个复核命令无需联网、GPU、模型权重或原始数据。它输出验证 JSON 和八组 PDF/PNG 图：学习曲线、分段长度分布、六个 fMRI domain 的层级图。

## 两种复现层级

1. **随包可以运行的汇总复核**：三组 full-dev FLOPs、学习曲线/AUC、四个英文模型的分段长度统计，以及六被试描述性 fMRI 图。这里只能验证保留的汇总数据及其计算，不等于从原始数据独立重复实验。
2. **需要自行提供输入的重新分析**：完整 dev 分段计数、checkpoint 分段审计、Hippocorpus 清洗/分词、从评估输出重新汇总学习曲线、真实被试配对统计。运行方法见 [英文 README](README.md)；数据格式和获取要求见 [DATA.md](docs/DATA.md)。

中文正式 full-dev 口径为 **89,909 条序列、2.946 tokens/concept、约 47.07 GF ideal forward FLOPs**。代码已包含 readout memory `C_B + T`，不要重复加算。理想长度处估计与 padding-aware 估计是不同量；后者中文约 59.91 GF，不能用前者推断训练或推理更快。

## 主要目录

- `src/babyconceptlm_analysis/`：FLOPs、统计检验、汇总复核、绘图和命令行入口。
- `scripts/`：可传入路径的分段分析、全 dev 计数、语料预处理、曲线汇总、固定 revision 下载、发布检查。
- `configs/`：六个 matched checkpoint 的公开名称、完整 revision、权重哈希，以及计算形状参数。
- `examples/reference/`：经过筛选的汇总参考数据。
- `examples/fmri_synthetic.csv`：**纯合成示例，不是论文证据**。
- `tests/`：基础测试；`tests/inference/` 为可选依赖相关测试。
- `docs/`：复现范围、数据来源、改编来源哈希及发布清单。

## 必须注意的边界

- English/Multilingual matched checkpoint 与 public-final submission 不是同一个配方，不能替换使用。
- Chinese matched pair 两者均无 DWA；TA1 对比联合改变完整 concept pathway 和层数分配，不是仅改变分段器的消融。
- 六被试 fMRI 层级图是另一个 unmatched 描述性分析，不是十二被试 matched paired comparison。
- 新的配对统计要求明确随机种子和 Holm 检验族。原论文 bootstrap 的随机实现信息不足，不能声称新生成区间逐位复现原区间。
- full-dev 重新分析的置换区间统一按每轮 `mean(F(C_B))` 计算；原脚本区间使用 `F(mean(C_B))`，差异已记入复现说明。保留的原论文点估计没有改动。
- 不随包提供原始 fMRI、真实逐被试分数、Hippocorpus 故事、训练语料、模型权重、tokenizer、凭据或实验日志。运行时的明细输出也不应直接公开。

## 上传前

```bash
python scripts/check_release.py
python scripts/check_release.py --zip dist/babyconceptlm-paper-analysis.zip
```

只把这个独立目录或生成 ZIP 的内容作为新仓库，不要把整个训练工程上传。工具不会执行 GitHub 上传。

**许可证尚待作者确认**，详见 [LICENSE_STATUS.md](LICENSE_STATUS.md)。完成许可证、第三方权利和数据授权复核后，再按 [发布清单](docs/RELEASE_CHECKLIST.md) 发布。目录结构可用于 GitHub，但目前不应宣称已完成开源授权。
