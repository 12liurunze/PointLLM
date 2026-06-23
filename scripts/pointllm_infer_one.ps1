$ErrorActionPreference = "Stop"

param(
  [string]$ObjectId = "05ba73f3f2bb4050988e087f53e98dc9",
  [string]$Question = "Describe this 3D object in detail."
)

$RepoRoot = Split-Path -Parent $PSScriptRoot
$EagleEyeRoot = Join-Path $RepoRoot "EAGLE_EYE"
$PointLLMRepo = "/root/autodl-tmp/pointLLM/pointllm"
$BaseModel = "F:\download\point7B"
$PointCloudData = "F:\download\8192_npy"
$HeadDir = "F:\download\pointllm_eagle_head"
$OutputJsonl = "F:\download\pointllm_ee_answers.jsonl"

Set-Location $EagleEyeRoot
$env:PYTHONPATH = "$PointLLMRepo;$EagleEyeRoot;$env:PYTHONPATH"

python -m eagle_eye.evaluation.gen_ee_answer_pointllm `
  --base-model-path $BaseModel `
  --ee-model-path $HeadDir `
  --pointllm-repo-path $PointLLMRepo `
  --data-path $PointCloudData `
  --object-id $ObjectId `
  --question $Question `
  --output-jsonl $OutputJsonl `
  --torch-dtype float16
