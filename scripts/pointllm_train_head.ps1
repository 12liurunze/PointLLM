$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$EagleEyeRoot = Join-Path $RepoRoot "EAGLE_EYE"
$PointLLMRepo = "C:\Users\lrz\PointLLM"
$BaseModel = "F:\download\point7B"
$DataDir = "F:\download\pointllm_eagle_data"
$HeadDir = "F:\download\pointllm_eagle_head"

Set-Location $EagleEyeRoot
$env:PYTHONPATH = "$PointLLMRepo;$EagleEyeRoot;$env:PYTHONPATH"

python -m eagle_eye.train.train_pointllm `
  --basepath $BaseModel `
  --pointllm-repo-path $PointLLMRepo `
  --tmpdir $DataDir `
  --cpdir $HeadDir `
  --bs 4 `
  --num-epochs 20 `
  --mixed-precision fp16
