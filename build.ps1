# --- FAST AND CLEAN BUILD SCRIPT FOR PYINSTALLER PROJECT ---
# Author: EmberNoGlow & ChatGPT
# -----------------------------------------------------------

# --- Configuration ---
$ReleaseDir = "ReleaseBuild"
$VenvName = ".venv"
$PyInstallerPath = Join-Path $VenvName "Scripts\pyinstaller.exe"
$PipPath = Join-Path $VenvName "Scripts\pip.exe"
$MainScript = "main.py"
$DistExecutable = "sdfeditor"
$GlfwDllSource = Join-Path $VenvName "Lib\site-packages\glfw\glfw3.dll"

# --- Step 0: Cleanup old build folders ---
Write-Host "0. Cleaning up old build folders..." -ForegroundColor Gray
foreach ($folder in "dist", "build", $ReleaseDir) {
    if (Test-Path $folder) {
        Remove-Item -Path $folder -Recurse -Force
    }
}

# --- Step 1: Create Release Directory ---
Write-Host "1. Creating release directory: $ReleaseDir..." -ForegroundColor Cyan
New-Item -Name $ReleaseDir -ItemType Directory -ErrorAction SilentlyContinue | Out-Null

# --- Step 2: Create or Verify Virtual Environment ---
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

# --- Step 3: Install Dependencies ---
Write-Host "3. Installing dependencies from requirements.txt..."
if (Test-Path "requirements.txt") {
    & $PipPath install -r requirements.txt
} else {
    Write-Warning "requirements.txt not found. Skipping dependency installation."
}

# --- Step 4: Ensure PyInstaller is Installed ---
Write-Host "4. Ensuring PyInstaller is installed..."
& $PipPath install pyinstaller

# --- Step 5: Run PyInstaller Build ---
Write-Host "5. Running PyInstaller build..."

# IMPORTANT:
# We do NOT add glfw3.dll via --add-binary to avoid DLL conflicts.
$PyCommand = @(
    "--onedir",
    "--name", $DistExecutable,
    "--windowed",
    $MainScript
)

& $PyInstallerPath @PyCommand

if ($LASTEXITCODE -ne 0) {
    Write-Error "PyInstaller failed (Exit Code $LASTEXITCODE)."
    exit 1
}

# --- Step 6: Move Compiled Executable and Dependencies ---
Write-Host "6. Copying built files to $ReleaseDir..."

$DistDir = ".\dist\$DistExecutable"

if (Test-Path $DistDir) {
    Copy-Item -Path "$DistDir\*" -Destination $ReleaseDir -Recurse -Force
    Write-Host "All runtime files (exe + dependencies + _internal) copied to $ReleaseDir."
} else {
    Write-Warning "Could not find build directory '$DistDir'."
}


# --- Step 7: Copy glfw3.dll ---
Write-Host "7. Copying glfw3.dll to $ReleaseDir..."
if (Test-Path $GlfwDllSource) {
    Copy-Item -Path $GlfwDllSource -Destination $ReleaseDir -Force
    Write-Host "glfw3.dll copied."
} else {
    Write-Warning "GLFW DLL not found in venv. Application may not start."
}

# --- Step 8: Copy Shaders Folder ---
Write-Host "8. Copying 'shaders' folder..."
if (Test-Path ".\shaders") {
    Copy-Item -Path ".\shaders" -Destination $ReleaseDir -Recurse -Force
    Write-Host "Shaders folder copied."
} else {
    Write-Warning "Folder 'shaders' not found. Skipping."
}

# --- Step 9: Copy GUI Fonts ---
Write-Host "9. Copying 'gui/fonts'..."
if (Test-Path ".\gui\fonts") {
    New-Item -Name "gui" -Path $ReleaseDir -ItemType Directory -ErrorAction SilentlyContinue | Out-Null
    Copy-Item -Path ".\gui\fonts" -Destination "$ReleaseDir\gui\" -Recurse -Force
    Write-Host "gui/fonts folder copied."
} else {
    Write-Warning "Folder 'gui/fonts' not found. Skipping."
}

# --- Step 10: Copy Documentation Files ---
Write-Host "10. Copying documentation files..."
foreach ($doc in "README.md", "LICENSE", "LICENSE.txt") {
    if (Test-Path $doc) {
        Copy-Item -Path $doc -Destination $ReleaseDir -Force
        Write-Host "Copied $doc."
    }
}

# --- Step 11: Create GitHub Shortcut ---
Write-Host "11. Creating GitHub URL shortcut..."
$UrlFilePath = Join-Path $ReleaseDir "Visit Github.url"
$UrlContent = @"
[InternetShortcut]
URL=https://github.com/EmberNoGlow/SDF-Model-Editor-Demo
"@
Set-Content -Path $UrlFilePath -Value $UrlContent -Encoding UTF8

# --- Step 12: Create ZIP Archive ---
Write-Host "12. Creating ZIP archive..."

$ZipFile = "SDFEditor.zip"
if (Test-Path $ZipFile) { Remove-Item $ZipFile -Force }

Compress-Archive -Path "$ReleaseDir\*" -DestinationPath $ZipFile -CompressionLevel Optimal

Write-Host "`nBUILD COMPLETED SUCCESSFULLY!" -ForegroundColor Green
