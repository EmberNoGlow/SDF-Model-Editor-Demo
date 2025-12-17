# SDF Model Editor (Demo)

![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)
![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)


## What is it?
This is a small project in which I created rendering and full interaction with sdf primitives. Using `Python, GLSL, Imgui, glfw, pyopengl`.

The cursor, co-pilot, and gpt chat helped me bring the idea to life. I admit, it wasn't as colorful as I imagined. See the "About" section below.

Despite all the difficulties, **I am happy with the result**.

---

## **Warning!**
This **work is in progress**, and this tool currently has limited functionality. See the **roadmap** below for plans. **Please support** the project's development!

---

|  |  |
| :---: | :---: |
| ![](Screenshots/Screenshot_1.png) | ![](Screenshots/Screenshot_2.png) |
| ![](Screenshots/Screenshot_3.png) | ![](Screenshots/Screenshot_4.png) |

---

## Contributing
If you find a bug or want to suggest an improvement, open a new issue or even Pull Request.

---

### Founded bugs:
1. Cone and Plane don't work.
2. When changing the rendering scale, the screen shifts.
3. And more, more small bugs...

---

## Roadmap

Goal: Create an MVP! But to achieve this, functionality is key. Look at the plans:

### Functionality (the main thing)
1. Save/Load function
2. Add export to 3d model formats (obj, gltf, etc.), using voxelizations and marching cubes.
3. Add the ability to drag and rotate primitives using gizmo.

---

### User Interface (The most insignificant)
1. Add themes.
2. Localization

...And more...

---

# About

## How did I come up with this?!
It's all because of this [article](https://iquilezles.org/articles/smin/)! I was inspired by creating a character using SDF, which makes creating a stylized character much easier. No sculpting! Just 10-20 primitives – and the model is ready.

## How I did it?
I decided to create an editor. But I only knew Python and Glsl. So I decided to build an MVP using AI. I used chat gpt, copilot and cursor, but it's not that simple. I was able to create the most difficult thing for me - the user interface and rendering on sdf.

But soon, I ran out of free space (a common occurrence). I realized the app was quite simple, but there was a lot of code! I spent several weeks studying and refactoring it. Then I learned about cursor. I submitted a request to him to create a more advanced editor. And he changed everything...

I reached my limits. I spent another month refining the code (or rather, studying it). The result was something... not perfect, not a polished product, and not very functional.

**I learned my lesson that AI isn't the best assistant** (or maybe I'm just a bad developer).

**But it was cool**. Studying and fixing code three times (there were a lot of errors!!!), which does roughly the same thing, was fascinating.

---

## Links
🔗 [wikipedia article](https://en.wikipedia.org/wiki/Signed_distance_function) about SDF


🔗 sdf [functions](https://iquilezles.org/articles/distfunctions)


🔗 [shadertoy](https://shadertoy.com)


🔗 [cursor ai](https://cursor.com/)
