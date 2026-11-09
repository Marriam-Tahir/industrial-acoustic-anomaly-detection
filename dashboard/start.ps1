param([switch]$Streamlit)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Create the environment and install requirements first. See README.md.'
}
if ($Streamlit) {
    & $projectPython -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502
} else {
    & $projectPython main.py
}
