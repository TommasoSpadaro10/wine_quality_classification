Param(
  [Parameter(Mandatory = $true, Position = 0)]
  [string]$Action
)

$pythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path $pythonExe) {
  & $pythonExe src/main.py --action $Action
} else {
  python src/main.py --action $Action
}