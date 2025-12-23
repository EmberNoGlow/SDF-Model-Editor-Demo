# Contributing to SDF Model Editor Demo 🎮

Thanks for wanting to contribute! This project is all about **interactive SDF (Signed Distance Field) modeling** with Python and GLSL. Whether you're fixing a bug, adding a new shape, or improving the UI - your help is welcome!

Here’s how to get started:

---

### 1️⃣ Fork & Clone the Repo
1. Click the **"Fork"** button at the top of the [repo page](https://github.com/EmberNoGlow/SDF-Model-Editor-Demo).
2. Clone your fork:
```bash
git clone https://github.com/your-username/SDF-Model-Editor-Demo.git
cd SDF-Model-Editor-Demo
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Create a Branch

```bash
git checkout -b your-feature-name  # e.g., "add-new-sdf" or "fix-camera-bug"
```

### 4️⃣ Make Your Changes
- For new SDF shapes: Add the GLSL function in shaders/sdf.glsl and update the UI in main.py.
- For bug fixes: Test thoroughly and describe the fix in your PR.
- For UI improvements: Include screenshots in your PR!

### 5️⃣ Test Your Changes
Run the editor to make sure everything works:

```bash
python main.py
```

### 6️⃣ Commit & Push

```bash
git add .
git commit -m "feat: add new SDF primitive"  # Use clear, concise messages
git push origin your-feature-name
```

### 7️⃣ Open a Pull Request (PR)
- Go to the repo.
- Click “New Pull Request” and select your branch.
- Describe your changes and why they’re useful.
- Bonus: Add a screenshot or GIF if it’s a visual change!

## 💡 Contribution Tips
- **Keep PRs small** (e.g., one feature or fix per PR).
- Follow the existing **code style**
- **Ask questions** - open an issue if you’re unsure!
- **Be kind** - we’re all learning and building together.

## 🙏 Thank You!
Your contributions **help** make this project better for everyone. Whether you’re adding a new SDF shape or fixing a bug, we appreciate your time and effort! 🌟
