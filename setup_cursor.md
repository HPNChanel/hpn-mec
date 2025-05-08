# 🧠 Cursor AI Setup for HPN Medicare

This file defines the full development context and Cursor behavior strategy for the **HPN Medicare** project.

---

## 📦 Project Overview

**HPN Medicare** is a smart healthcare management system with three main components:

1. **Frontend (`edutech_ui/`)**
   - ReactJS + MUI + Vite
   - Modular structure using components, hooks
   - Features: Auth (login/register), Dashboard, Health Record Viewer, AI Inference Viewer, Dark/Light mode

2. **Backend (`edutech_api/`)**
   - FastAPI-based RESTful backend
   - MySQL database (users, health_records, metrics, logs)
   - JWT-based auth system, role-based access
   - Inference endpoints interact with `ai_module/`

3. **AI Module (`ai_module/`)**
   - Includes anomaly detection models (AutoEncoder, Isolation Forest)
   - Trained using PyTorch, served via FastAPI
   - Outputs: latent vectors, reconstruction errors, anomaly scores

---

## 🎯 Cursor Objective

Cursor Pro is expected to:
- Generate code for both backend and frontend modules
- Explain/refactor/document complex or long blocks of code
- Write unit tests (e.g. `pytest`, `unittest`)
- Select the most appropriate interaction mode (Agent / Ask / Manual)
- Act intelligently in context of AI pipeline and data-driven features

---

## 🧭 Behavior by Mode

| Mode | When to Use | Example Prompt |
|------|-------------|----------------|
| `Agent` | Use when task is clear and execution is safe | `// Create a responsive login form using MUI, with validation` |
| `Ask` | Use when task has multiple implementation paths or affects architecture | `// Suggest 2 ways to show anomaly explanation in React` |
| `Manual` | Use when developer wants control over output or is reviewing/refactoring | `// Refactor this function for clarity and performance` |

---

## 🛠 Cursor Guidelines

- Use React functional components and MUI best practices.
- Always make API calls async, well-typed, and follow REST.
- Avoid suggesting code that breaks structure or duplicates helpers.
- Always document important functions with inline comments or docstrings (unless explicitly told not to).
- Don’t suggest frontend mock data – expect real API responses.

---

## 🚀 Tips for Developer (Huỳnh Phúc Nguyên)
- Use `Ctrl + .` to quickly switch between modes.
- Use comment commands: `// Explain`, `// Refactor`, `// Test`, `// Improve`, `// Optimize`.
- Keep Claude AI Pro for high-level logic design, and let Cursor do the implementation/refactor.
- Combine Copilot Pro for autocompletion + Cursor for understanding & generation.

---

## ✅ Example Cursor Prompts

```tsx
// React: Create a 3-step form for health input using MUI Stepper, validate each step
# FastAPI: Write an endpoint that retrieves health records by user_id and date range from MySQL

# Test: Write pytest for get_anomaly_score function using 3 typical cases