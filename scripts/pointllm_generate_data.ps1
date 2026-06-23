$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$EagleEyeRoot = Join-Path $RepoRoot "EAGLE_EYE"
$PointLLMRepo = "/root/autodl-tmp/pointLLM"
$BaseModel = "/root/autodl-tmp/point7B_v1.1"
$PointCloudData = "/root/autodl-tmp/pointLLM/data/objaverse_data"
$Annotation = "/root/autodl-tmp/pointLLM/data/anno_data/PointLLM_brief_description_660K_filtered.json"
$OutputDir = "/root/autodl-tmp/pointllm_eagle_data"

Set-Location $EagleEyeRoot
$env:PYTHONPATH = "$PointLLMRepo;$EagleEyeRoot;$env:PYTHONPATH"

python -m eagle_eye.ge_data.get_data_all_pointllm `
  --base-model-path $BaseModel `
  --pointllm-repo-path $PointLLMRepo `
  --data-path $PointCloudData `
  --anno-path $Annotation `
  --outdir $OutputDir `
  --index 0 `
  --start 0 `
  --end 10000 `
  --conversation-types "single_round,multi_round,detailed_description" `
  --torch-dtype float16
