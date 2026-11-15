# Diffusion-Based Image Editing

使用预训练 Stable Diffusion 1.5，自主实现 DDIM 反演、采样、CFG 与 mask-guided
latent blending 的计算机视觉课程项目。**当前仅 Phase 0：工程初始化。尚不能编辑图片。**
所有未来修改须遵循 [AGENTS.md](AGENTS.md)。

## 安装（PowerShell，仓库根目录）

建议 Python 3.11；支持范围为 3.10–3.12。开发与模型依赖隔离：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
python scripts/check_environment.py --output outputs/environment.json
```

Phase 0 离线测试无需 torch、GPU 或模型。仅安装 PyYAML、pytest 和构建工具。
项目元数据安装不代表完整模型依赖已验证。

模型运行环境选择固定的 torch/torchvision 配对和成熟版本库，不追逐最新版。
本机 NVIDIA GPU 的候选安装命令（会下载大型运行库，尚未执行）：

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/check_environment.py --output outputs/environment-model.json
```

CPU 环境将 `cu124` 换成 `cpu`。CPU 只建议轻量测试，完整模型加载/生成很慢且需要
较多 RAM。PyTorch CUDA wheel 自带运行库，无须为推理另行安装完整 CUDA Toolkit；
驱动和实际 CUDA 可用性以检查脚本为准。

固定版本：torch 2.5.1、torchvision 0.20.1、diffusers 0.35.1、transformers 4.44.2、
accelerate 1.0.1、safetensors 0.6.2、huggingface-hub 0.34.4、numpy 1.26.4。
这是候选兼容基线；真实 GPU/模型兼容性仍需集成验证。直接依赖已固定，完整传递依赖
锁定应在模型环境安装并验证后生成，不把未经验证的解析结果冒充 lockfile。
Gradio、OpenCV、绘图和指标库在对应阶段再引入。

## 模型与 smoke checks

- ID: `stable-diffusion-v1-5/stable-diffusion-v1-5`
- Revision: `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`
- [官方模型卡](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
  和 [访问/文件元数据](https://huggingface.co/api/models/stable-diffusion-v1-5/stable-diffusion-v1-5)：
  2026-11-15 检查为公开、ungated，采用 CreativeML OpenRAIL-M；使用和分发须遵守模型许可。
- Diffusers 格式，包含 CLIP、VAE、4-channel U-Net、scheduler 和 FP16 safetensors。
  只支持 epsilon-prediction base model；不把 inpainting checkpoint 当作兼容替代。
- 权重需要数 GB 下载和缓存空间；预留约 10 GB 磁盘。RTX 4080 Laptop 12 GB
  可作为 512×512 / batch=1 / CUDA FP16 的起点，实际峰值须测量。
- U-Net/CLIP 用自动设备精度，VAE 用 FP32。缓存位于 `.cache/huggingface`。

默认配置严格离线。以下操作只有在依赖与模型缓存均存在时才能成功：

```powershell
.\.venv\Scripts\python.exe scripts/check_model.py --forward
.\.venv\Scripts\python.exe scripts/test_generation.py --steps 20
```

经用户确认后，首次下载可显式添加 `--allow-download`。不要把此选项用于单元测试。
`check_model.py` 验证组件加载、文本编码和一次 U-Net forward；
`test_generation.py` 是**内置 text-to-image 参考 smoke**，保存
`outputs/generation/reference.png`，不是最终编辑算法。它保留内置 safety checker。
两者默认执行不下载；若自行在配置中关闭 `local_files_only`，会允许联网。

## 架构与后续使用

```text
configs/                  显式 YAML 配置
src/diffusion_editor/
  config.py               配置验证
  models/sd_backend.py    预训练组件加载（当前实现）
  diffusion/              后续：schedule、inversion、sampler、CFG
  editing/                后续：mask、blending、editor
  pipelines/              后续：重建和编辑编排
  evaluation/             后续：指标和可复现实验
  demo/                   后续：Gradio
  utils/                  环境诊断和日志
scripts/                  环境检查、组件 forward、参考生成
tests/                    离线 CPU 单元测试
docs/                     架构、数学约定、路线和实验记录
assets/examples/          后续输入样例（记录来源/使用权限）
outputs/                  运行产物，不入版本控制
```

Demo、编辑命令及实验命令尚未实现，不能在此阶段运行。
后续顺序与验收见 [roadmap](docs/roadmap.md)；本轮结束后等待 Phase 1 指令。

## 已知限制

当前没有模型权重、PyTorch 安装或真实推理结果。离线 mock 只证明加载接口契约，
不证明 CUDA、真实 checkpoint 或生成质量。DDIM 反演本身是近似过程，VAE 有重建误差；
latent mask 不保证像素严格不变，后续会将最终像素合成独立配置。
完整项目完成状态以 AGENTS.md 的 definition of done 为准。
