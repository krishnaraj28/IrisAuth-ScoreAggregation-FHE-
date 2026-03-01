## HomoIris — biometric authentication system with Fully Homomorphic Encryption

**HomoIris** is a privacy-preserving biometric authentication system that leverages **Fully Homomorphic Encryption (FHE)** to guarantee end-to-end protection of biometric feature data. Built using **TenSEAL’s CKKS scheme**, the system performs similarity computations directly on encrypted vectors, ensuring that sensitive biometric information is never exposed in plaintext.

---

### 🔐 Core Concept

The system is designed so that:

* Biometric templates remain encrypted at all times
* Matching occurs only on ciphertexts
* The server never accesses raw biometric data
* Final decisions are made client-side after decryption

This architecture ensures **strong privacy, security, and regulatory compliance** for biometric authentication.

---

### 🧾 Enrollment Process

During enrollment:

* Feature vectors are **L2-normalized**
* Each user receives a **unique CKKS keypair**
* Encryption parameters:

```
poly_modulus_degree = 8192
coeff_mod_bit_sizes = [60, 40, 40, 60]
global_scale = 2^40
```

* Encrypted templates are stored alongside the user’s public context in the secure database.

---

### 🔎 Authentication Workflow

1. Query features are normalized
2. Query is encrypted using the target user's public key
3. Server computes **cosine similarity homomorphically**
4. Similarity scores are produced for each stored template
5. Scores are aggregated
6. Client decrypts final results and makes the decision

At no stage does the server decrypt biometric information.

---

### 📊 Score-Level Aggregation

To improve reliability, the system compares the query against **multiple enrolled templates** and aggregates the encrypted similarity scores.

Supported aggregation logic:

* Encrypted averaging
* Multi-template comparison
* Ciphertext-domain operations only

This improves robustness against real-world noise such as:

* eye movement
* blinking
* lighting variation

---

### ⚖️ Dual Decision Strategy

Authentication is determined using two complementary checks:

**1. Rule-Based Verification**

* Requires at least one feature similarity ≥ **0.999**
* Must occur across a minimum number of templates

**2. Traditional Threshold Verification**

* Requires average similarity ≥ **0.70**

Final decision = combination of both methods.

---

### 🧠 Explainable AI (XAI) Layer

The system integrates a full **decision transparency module** that provides:

* Per-template similarity scores
* Feature-level confidence analysis
* Near-miss detection
* Confidence evaluation
* Recommended actions

📄 **Export Options**

* JSON reports
* PDF reports with visual analytics

This ensures **auditability, interpretability, and trustworthiness**.

---

### 🚀 Key Advantages

* Fully encrypted biometric matching
* Zero plaintext exposure
* Multi-template robustness
* Client-controlled decisions
* Explainable authentication logic
* Exportable verification reports
* Scalable architecture

---

### 🧩 System Architecture Philosophy

> Privacy + Cryptography + Explainability

HomoIris demonstrates that biometric systems can be:

* secure
* transparent
* accurate
* privacy-preserving

without compromising usability or performance.

---

### 📌 Summary

**HomoIris** combines homomorphic encryption, multi-template similarity aggregation, and explainable AI to deliver a next-generation biometric authentication system that is secure by design, transparent by default, and reliable in real-world conditions.

