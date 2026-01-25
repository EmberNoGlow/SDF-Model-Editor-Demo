# --- CONVENIENT AND FAST SCRIPT THAT COMPILES A PROJECT WITH JUST ONE COMMAND ---
# Autor: EmberNoGlow & ChatGPT
# -----------------------------

# --- Configuration ---
$ReleaseDir = "ReleaseBuild"
$VenvName = ".venv"
$PyInstallerPath = Join-Path $VenvName "Scripts\pyinstaller.exe"
$PipPath = Join-Path $VenvName "Scripts\pip.exe"
$MainScript = "main.py"
$GlfwDllSource = Join-Path $VenvName "Lib\site-packages\glfw\glfw3.dll"
$DistExecutable = "sdfeditor" # Name used by PyInstaller --name

# --- Step 1: Create Release Directory ---
Write-Host "1. Creating release directory: $ReleaseDir..." -ForegroundColor Cyan
New-Item -Name $ReleaseDir -ItemType Directory -ErrorAction SilentlyContinue | Out-Null

# --- Step 2: Create Virtual Environment ---
Write-Host "2. Checking and creating virtual environment $VenvName..."
if (-not (Test-Path $PipPath)) {
    Write-Host "Creating environment via python -m venv..."
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        Write-Error "Python command not found. Ensure Python is in PATH."
        exit 1
    }
    & python -m venv $VenvName
    if ($LASTEXITCODE -ne 0) { Write-Error "Failed to create venv."; exit 1 }
} else {
    Write-Host "Virtual environment already exists." -ForegroundColor DarkGray
}

# --- Steps 3 & 4: Install Dependencies ---
Write-Host "3 & 4. Installing dependencies from requirements.txt..."
if (Test-Path "requirements.txt") {
    & $PipPath install -r requirements.txt
} else {
    Write-Warning "requirements.txt not found. Skipping dependency installation."
}

# --- Step 5: Install PyInstaller ---
Write-Host "5. Ensuring PyInstaller is installed..."
& $PipPath install pyinstaller

# --- Step 6: Execute PyInstaller Build ---
Write-Host "6. Running PyInstaller build..."

$PyCommand = @(
    "--onefile",
    "--name", $DistExecutable,
    "--windowed",
    # Explicitly add DLL, copying it to the root of the distribution folder ('.')
    "--add-binary", "$GlfwDllSource;.",
    $MainScript
)

& $PyInstallerPath @PyCommand

if ($LASTEXITCODE -ne 0) {
    Write-Error "PyInstaller failed (Exit Code $LASTEXITCODE)."
    exit 1
}

# --- Step 7: Move Compiled Executable ---
Write-Host "7. Moving compiled executable to $ReleaseDir..."
$CompiledFile = if (Test-Path ".\dist\$($DistExecutable).exe") { ".\dist\$($DistExecutable).exe" } 
                else { ".\dist\$($DistExecutable)" }

if (Test-Path $CompiledFile) {
    Move-Item -Path $CompiledFile -Destination $ReleaseDir -Force
    Write-Host "Executable moved successfully."
} else {
    Write-Warning "Could not find compiled executable in 'dist'."
}

# --- Step 8 (NEW): Explicitly Copy GLFW DLL ---
Write-Host "8. Explicitly copying glfw3.dll to $ReleaseDir root..."
if (Test-Path $GlfwDllSource) {
    Copy-Item -Path $GlfwDllSource -Destination $ReleaseDir -Force
    Write-Host "glfw3.dll copied."
} else {
    Write-Warning "GLFW DLL not found at source path. Skipping explicit copy."
}

# --- Step 9 (was 8): Copy Shaders Folder ---
Write-Host "9. Copying 'shaders' folder to $ReleaseDir..."
if (Test-Path ".\shaders") {
    Copy-Item -Path ".\shaders" -Destination $ReleaseDir -Recurse -Force
    Write-Host "Shaders folder copied."
} else {
    Write-Warning "Folder 'shaders' not found. Skipping."
}

# --- Step 10 (NEW): Copy Fonts Folder ---
Write-Host "10. Copying 'gui/fonts' to $ReleaseDir/gui/..."
if (Test-Path ".\gui\fonts") {
    # Ensure the parent directory for fonts exists in ReleaseBuild
    New-Item -Name "gui" -Path $ReleaseDir -ItemType Directory -ErrorAction SilentlyContinue | Out-Null
    Copy-Item -Path ".\gui\fonts" -Destination "$ReleaseDir\gui\" -Recurse -Force
    Write-Host "gui/fonts folder copied."
} else {
    Write-Warning "Folder 'gui/fonts' not found. Skipping."
}

# --- Step 11: Create URL Shortcut ---
Write-Host "11. Creating URL shortcut 'Visit Github.url' in $ReleaseDir..."
$UrlFilePath = Join-Path $ReleaseDir "Visit Github.url"
$UrlContent = @"
[InternetShortcut]
URL=https://github.com/EmberNoGlow/SDF-Model-Editor-Demo
"@

Set-Content -Path $UrlFilePath -Value $UrlContent -Encoding UTF8

Write-Host "`nBUILD COMPLETED SUCCESSFULLY!" -ForegroundColor Green