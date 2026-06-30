---
name: Competition Submission
about: Use this template to submit your system for GenSIE 2026.
title: "[SUBMISSION] SEsml — Schema-guided Extraction with Small Language Models"
labels: submission
assignees: apiad
---

## 🏆 Submission Details

- **Official Team Name:** SEsml — Schema-guided Extraction with Small Language Models
- **Private Repository URL:** https://github.com/alejbv/SEsml.git
- **Contact Email:** alejandro.beltran@rect.uh.cu

## 👥 Participants

Please list all members of the team and their respective institutions.

| Name | Institution |
| :--- | :--- |
| Alejandro Beltrán | Facultad de Matemática y Computación, Universidad de La Habana |
| Daniel Toledo | Facultad de Matemática y Computación, Universidad de La Habana |

## 🚀 System Overview

Our approach is based on a **two-phase draft-then-decode** pipeline designed specifically for Small Language Models (<14B parameters). Instead of forcing the SLM to produce structured JSON directly — which causes timeout issues and trajectory bias under constrained decoding — we split the task:

1. **Phase 1 — DraftEngine (`draft.py`):** The SLM generates a semi-structured draft in a freeform `campo: valor` (field: value) format, wrapped in `<Thinking>…</Thinking><Draft>…</Draft>` tags. This allows the model to reason and extract without the overhead of JSON grammar constraints.

2. **Phase 2 — Constrained Decoding (`decoder.py`):** A second call to the SLM maps the draft values into the exact target JSON schema using `response_format=json_schema`. A local fallback parses `campo: valor` lines directly if structured output fails.

3. **Schema Optimization (`optimizer.py`):** A preprocessing step flattens `$ref`, removes `default`, enforces `additionalProperties: false`, and marks all fields `required` to minimize decoder state explosions.

The core innovation is the **prompt reduction and refinement** applied iteratively to fit within SLM context windows while maximizing extraction accuracy:

- **v3b prompt** (~1,163 chars / ~387 tok) with 3 explicit critical rules targeting the most frequent error patterns observed across 149 dev tasks:
  - *[1] VALORES EXACTOS* — Forces literal text copying, exact enum matching, and strict null semantics (no inference of optional fields like `severity_level`, `unit`).
  - *[2] INFERENCIA CONDICIONAL* — Reads the schema field `description` to determine whether inference is required (e.g., `dietary_tags`) or forbidden.
  - *[3] ARRAYS* — Short labels (1-4 words) for features/pros/cons, exhaustive scanning for long lists, and order preservation for sequences.

All previous prompt versions are preserved as comments in source for reproducibility.

### Pipelines

We submit **three pipelines** for evaluation:

1. **extraction:** SchemaOptimizer + DraftEngine (borrador `campo: valor`) + Constrained Decoding (2 fases). Prompt v3b con 3 reglas críticas para SLMs. **Pipeline principal para evaluación.**

2. **hybrid_cot:** SchemaOptimizer + llamada única con CoT visible (`<Thinking>`) + JSON directo vía `_extract_json`. Alternativa ligera al pipeline de 2 fases que elimina la llamada de constrained decoding a costa de precisión en esquemas complejos.

3. **adaptive:** SchemaAnalyzer + PromptAssembler + llamada única. Prompt adaptativo mediante similitud semántica sobre vectores de tareas de desarrollo (`data/dev_vectors.json`). El prompt se ensambla módulo por módulo según la complejidad, tipos de campo, dominio y patrones de error conocidos.

---

**Note:** By submitting this issue, you agree to grant read access to the repository to the organizers for evaluation purposes. Ensure your repository contains a valid `Dockerfile` in the root.
