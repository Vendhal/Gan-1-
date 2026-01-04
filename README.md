## 📌 Project-1: Synthetic Image Generator using Vanilla GAN

**(Integrated End-to-End System)**

### 🔹 Overview

This project integrates all individual modules into a **single, working end-to-end Vanilla GAN system** covering:

* Data preparation & preprocessing
* GAN model design
* Training & inference
* Deployment
* Monitoring & logging

All team members contributed their respective modules. The final system was **connected, stabilized, and future-proofed** into a unified pipeline suitable for real-world privacy-preserving image generation, as required in the Project-1 specification.

---

## 👥 Team Contributions (Module-wise)

### **Module 1 – Data Pipeline & Preprocessing**

**ADIRALA THENEESHA**

* Dataset organization and preprocessing pipeline
* Image normalization and formatting for GAN input
* Config-driven preprocessing using YAML
* Dataset structure designed for scalability and future reuse

---

### **Module 2 – Model Design (Vanilla GAN Architecture)**

**KAMBHAMPATI SAI SANDEEP**

* Vanilla GAN architecture (Generator & Discriminator)
* Loss formulation and optimizer configuration
* Model interfaces designed for clean training and inference separation
* Code structured for reproducibility and extensibility

> **Note:**
> Although the specification mentions Keras as an example, **PyTorch was used instead** because:
>
> * PyTorch is industry-standard for GAN research and experimentation
> * It offers clearer control over training, inference, and deployment
> * The underlying Vanilla GAN principles remain identical and fully compliant with the specification

---

### **Module 3 – Model Evaluation**

**MALLINA SRI SAI LOVA TEJENDRA**

* Evaluation logic for generated images
* Performance tracking hooks
* Code structured to allow future metrics such as FID or diversity scores

---

### **Module 4 – Optimization & Improvements**

**KARNATI SIVA SAI REDDY**

* Architectural refinements and performance considerations
* Training stability improvements
* Code prepared for future tuning and experimentation

---

### **Module 5 – Deployment Layer**

**KAMBHAMPATI SAI SANDEEP**
*(taken over due to teammate **AAKASH GUMMADI** being unavailable; faculty was informed in advance)*

* Streamlit-based deployment interface
* End-to-end inference deployment
* Clean `src/` package structure for execution
* Production-ready inference flow (model loading, caching, normalization)

---

### **Module 6 – Monitoring & Logging**

**KAVALA SAI VENKATA SURYANARAYANA**

* CSV-based logging of inference metadata
* Monitoring of:

  * Inference latency
  * Request count
  * Failure cases
* Code structured for future monitoring extensions

---

## 🔗 Integration & Enhancements

* Integrated all modules into **one runnable pipeline**
* Introduced a clean **`src/` package architecture**
* Added:

  * Monitoring dashboard (Streamlit)
  * Inference logging & reports
* Improved folder structure, execution reliability, and maintainability
* Applied **future-proofing practices** across modules (config files, modular design, versioning support)

---

## ✅ Notes for Evaluation

* All **core requirements of Project-1 are satisfied**
* Vanilla GAN principles strictly followed
* Choice of PyTorch is **implementation-level**, not conceptual deviation
* Additional features (dashboard, structured packaging, monitoring) go **beyond minimum specifications**
* Contributions are clearly documented with transparent ownership
* Faculty was informed in advance about module redistribution due to absenteeism
