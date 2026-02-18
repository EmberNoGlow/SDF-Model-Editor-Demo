# **SDF Model Editor (Demo)**
*A lightweight editor for creating and manipulating Signed Distance Field (SDF) primitives using Python, GLSL, and OpenGL.*


[![Python 3.11](https://img.shields.io/badge/python-3.11-orange.svg)](https://www.python.org/downloads/release/python-3110/) [![GLSL 330](https://img.shields.io/badge/GLSL-330-darkblue.svg)](https://www.khronos.org/registry/OpenGL/index_gl.php) [![Imgui 2.0.0](https://img.shields.io/badge/Imgui-2.0.0-red.svg)](https://github.com/pyimgui/pyimgui) [![GLFW 2.1.0](https://img.shields.io/badge/GLFW-2.1.0-yellow.svg)](https://www.glfw.org/) [![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![GitHub Issues](https://img.shields.io/github/issues/EmberNoGlow/SDF-Model-Editor-Demo)](https://github.com/EmberNoGlow/SDF-Model-Editor-Demo/issues) [![GitHub Stars](https://img.shields.io/github/stars/EmberNoGlow/SDF-Model-Editor-Demo?style=social)](https://github.com/EmberNoGlow/SDF-Model-Editor-Demo/stargazers)

---

![](Screenshots/Cover.png)

---


## **🌟 Overview**
This project is a **work-in-progress** editor for designing 3D models using **Signed Distance Fields (SDFs)**. Built with `Python`, `GLSL`, `ImGui`, `GLFW`, and `PyOpenGL`, it allows real-time rendering and interaction with SDF primitives.

While the current version is a **demo with limited functionality**, it serves as a foundation for a more powerful tool. The goal? To simplify 3D modeling by combining primitives into stylized characters or objects – **no sculpting required!**

> *I’m happy with the result, but there’s so much more to build!*

---

## **📌 Screenshots**
| ![Screenshot 1](Screenshots/Screenshot_1.jpg) | ![Screenshot 2](Screenshots/Screenshot_2.jpg) | ![Screenshot 2](Screenshots/Screenshot_3.jpg)
|:--------------------------------------------:|:--------------------------------------------:|:--------------------------------------------:|
| ![Screenshot 3](Screenshots/Screenshot_4.jpg) | ![Screenshot 4](Screenshots/Screenshot_5.jpg) | ![Screenshot 2](Screenshots/Screenshot_6.jpg)

---

## **⚠️ Current Status**
This is a late-stage prototype. I'm on the cusp of a major transition: the MVP is progressing well, and **[Roadmap](#-roadmap)** is nearing completion. Next, I'll move on to building the full product. Once the MVP is complete, I'll immediately focus on promoting the project on social media—and your support can make this launch truly meaningful!


> **Your feedback and contributions are welcome!** Open an [issue](https://github.com/EmberNoGlow/SDF-Model-Editor-Demo/issues) or submit a [PR](https://github.com/EmberNoGlow/SDF-Model-Editor-Demo/pulls).

---

## **🚀 Features**
✅ **Real-time SDF rendering** (GLSL shaders)

✅ **Cycles Additional rendering mode – Realistic Ray Tracing algorithm**

✅ **Smooth logic operations** (substract, intersect, union, etc.)

✅ **Multi-primitive composition** (spheres, boxes, cones, etc.)

✅ **ImGui-based UI** for intuitive controls

✅ **Save & Load** functions (using tkinter)

✅ **Export to 3d** using scikit-image

✅ **Built-in real-time scene code editor in GLSL** using Tkinter

---

## **🎯 Roadmap**

### *Current Progress*
- 📝 **Customization**
- 🧷 **UX improvements**
- 🐛 **Fix bugs**
- 🔨 Making an **GLSL code editor**
- 🌐 **Localization** - Preparation

### **Core Functionality (MVP Goals)**
- [x] **Free movement of the camera in 3d space**
- [x] **Translation operations for primitives** – rotation, scale, and position – can be changed in real time in the inspector.
- [x] **Save/Load** using JSON
- [x] **Undo/Redo support**
- [x] **Export to 3D formats** (OBJ)
- [ ] **Gizmo-based manipulation** (drag, rotate, scale primitives)
- [ ] **Write Docs!**

### **User Experience (Future Improvements)**
- [x] **Themes**
- [ ] **Customization**
- [ ] **Localization (i18n)**
- [ ] **Performance optimizations**

---

## **💞 Contributing**
Contributions are **highly encouraged**! Here’s how you can help:
1. **Report bugs** → Open an [issue](https://github.com/EmberNoGlow/SDF-Model-Editor-Demo/issues).
2. **Suggest features** → Share your ideas in [Discussions](https://github.com/EmberNoGlow/SDF-Model-Editor-Demo/discussions/).
3. **Submit code** → Fork the repo and create a [Pull Request](https://github.com/EmberNoGlow/SDF-Model-Editor-Demo/pulls).
4.  You can also send me **[DM on bsky](https://bsky.app/profile/embernoglow.bsky.social)** - report bugs, suggest features, or just Review!

> **Note:** This project is a learning experience – expect refactoring and experimentation!

---

## **💡 The Story Behind This Project**
### **Inspiration**
It all started with [this article](https://iquilezles.org/articles/smin/) on SDF-based modeling. The idea of creating stylized 3D characters with just **10-20 primitives** was too exciting to ignore!

### **The Journey**
- **Phase 1:** Built an MVP using AI tools (ChatGPT, Copilot, Cursor).
- **Phase 2:** Hit limitations – free chat limits, code complexity, and bugs.
- **Phase 3:** Spent **2 months** refactoring, learning, and improving.

> *AI isn’t a magic solution – it’s a tool. The real work is in understanding and refining the code.*

### **Lessons Learned**
✔ **AI accelerates prototyping** but requires deep debugging.

✔ **Refactoring is part of the process** – embrace it!

✔ **Small steps lead to big results** – this is just the beginning.

---

## **🔗 Resources**
- [SDF Wikipedia](https://en.wikipedia.org/wiki/Signed_distance_function)
- [Inigo Quilez’s SDF Functions](https://iquilezles.org/articles/distfunctions/)
- [Shadertoy](https://www.shadertoy.com/) (for SDF inspiration)
- [Cursor AI](https://cursor.com/) (the AI assistant that helped)

---

## **📜 License**
This project is licensed under the **MIT License** – see [LICENSE](LICENSE) for details.

---

## **🙌 Acknowledgments**
- **Inigo Quilez** for SDF research.
- **AI tools** (ChatGPT, Copilot, Cursor) for assistance.
- **Open-source community** for inspiration and libraries.

---

## **🚀 Ready to Try It?**
### Download compiled
Compiled executables for windows can be **found in [releases](https://github.com/EmberNoGlow/SDF-Model-Editor-Demo/releases)**


> *NOTE*: I don't have the ability to compile a linux app yet.


### Clone and run main.py
```bash
git clone https://github.com/EmberNoGlow/SDF-Model-Editor-Demo.git
cd SDF-Model-Editor-Demo
python -m venv .venv
.venv/Scripts/Activate.ps1
pip install -r requirements.txt
python main.py
```

> *NOTE:* To install the imgui package you will need a c++ compiler. My advice is to use **[mingw](https://sourceforge.net/projects/mingw/)** – if you **only** need a compiler for this task. **[Build Tools](https://visualstudio.microsoft.com/downloads/?q=build+tools)** will do for this task too, but it **weighs more**.


### Compilation
You have several options for compilation: automatic and manual. Automatic compilation is **highly recommended** as it will save a lot of time.

#### Automatic
The automated process is as simple as possible:
1. Make sure you are in the directory where the project is located.
2. Run the PowerShell script:
➤ `.\build.ps1`
3. Wait for the compilation to complete (the console should display **BUILD COMPLETED SUCCESSFULLY!**). You now have an executable file in the ReleaseBuild folder.

This method has been **tested on Windows**.


#### Manually

1. **First, install pyinstaller.**

```bash
pip install pyinstaller
```

2. **Compile into one file**

```bash
pyinstaller --onedir --name sdfeditor --windowed main.py
```

> *NOTE*: You should make sure that you have a virtual environment named **.venv created and glfw installed** (it is added to requirements). If there is none or the **path is incorrect, specify the correct path** to `the glfw3.dll` file located in the virtual environment in the `\Lib\site-packages\glfw folder`!

3. **Move the compiled file to a separate folder and drag the shaders folder to this directory. You should get it like this:**

```

root directory
├── gui/fonts
│    └── *.ttf files
│
├── shaders
│    ├── fragment
│    │    ├── cycles.glsl
│    │    └── template.glsl
│    │
│    ├── sdf_library.glsl
│    └── vertex_shader.glsl
│
├── glfw3.dll
└── sdfeditor.exe

```


---

### **Final Thoughts**
This project is a **testament to experimentation** – imperfect, evolving, but full of potential. Whether you’re here to **learn, contribute, or build**, I’m excited to see where it goes next!

**Let’s create something amazing together.** 🚀

---

## Support

Your support helps development! Help by leaving a review, reporting an error, suggesting an idea in Discussions, creating a Pull Request, or just **star ⭐** the repository!.


## Follow me

<a href="https://dev.to/embernoglow" target="_blank"><img src="https://img.shields.io/badge/Dev.to-black?style=flat-square&logo=dev.to&logoColor=white" alt="Dev.to" width="13%" /></a><a href="https://bsky.app/profile/embernoglow.bsky.social" target="_blank"><img src="https://img.shields.io/badge/Blue_Sky-1DA1F2?style=flat-square&logo=bluesky&logoColor=white" alt="Blue Sky" width="13%" /></a><a href="https://github.com/EmberNoGlow" target="_blank"><img src="https://img.shields.io/badge/GitHub-black?style=flat-square&logo=github&logoColor=white" alt="GitHub" width="13%" /></a>
