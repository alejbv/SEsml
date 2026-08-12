## 🏆 Submission Details

- **Official Team Name:** SEsml — Schema-guided Extraction with Small Language Models
- **Private Repository URL:** https://github.com/alejbv/SEsml.git
- **Contact Email:** alejandro.beltran@rect.uh.cu

## 👥 Participants

| Name | Institution |
| :--- | :--- |
| Alejandro Beltrán | Facultad de Matemática y Computación, Universidad de La Habana |

## 🚀 System Overview

We submit three pipelines for evaluation, all sharing a **SchemaOptimizer** that flattens `$ref`, enforces `additionalProperties: false`, marks all fields `required`, strips `default`, and normalizes schemas for constrained decoding engines.

All pipelines use a compact **v3b prompt** (~1,163 chars / ~387 tok) refined through iterative ablation across 149 dev tasks, with three critical rules targeting the most frequent error patterns:
- **[1] VALORES EXACTOS** — literal copying, exact enum matching, strict null semantics
- **[2] INFERENCIA CONDICIONAL** — schema field descriptions determine when inference is allowed
- **[3] ARRAYS** — short labels (1-4 words), exhaustive scanning, order preservation

### Pipelines

1. **extraction** (Principal): SchemaOptimizer → DraftEngine (unconstrained `campo: valor` draft) → Constrained Decoder (`json_schema`). Two-phase draft-then-decode architecture that decouples semantic reasoning from structural enforcement, mitigating the *constraint tax* on SLMs.

2. **hybrid_cot**: SchemaOptimizer → single call with visible CoT (`<Thinking>…</Thinking>`) + `_extract_json`. Lighter alternative that skips the second decoding call at the cost of precision on complex schemas.

3. **adaptive**: SchemaAnalyzer → PromptAssembler → single call. Prompt assembled per task via semantic similarity over dev task vectors (`data/dev_vectors.json`), adapting modules by complexity, field types, domain, and known error patterns.

Detailed system description available in our working notes paper: `paper/gensie2026-sesml.pdf`

---

## ✅ Bookkeeping

- **✅ Paper sent** — the final camera-ready version ([paper/gensie2026-sesml.pdf](https://github.com/alejbv/SEsml/blob/main/paper/gensie2026-sesml.pdf)), following CEUR author instructions.
- **✅ Signed CEUR agreement sent** — as a PDF scan ([paper/ceur_signed.pdf](https://github.com/alejbv/SEsml/blob/main/paper/ceur_signed.pdf)).
- **🎟️ Attendance:** No — I do not plan to attend the event in person.
- **🔎 Paper reviewed:** Yes — I have checked the overview and my team/results are represented correctly.

---

**Note:** By submitting this issue, you agree to grant read access to the repository to the organizers for evaluation purposes. Ensure your repository contains a valid `Dockerfile` in the root.
