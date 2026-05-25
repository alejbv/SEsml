"""
Prompt templates for the Extraction v2 pipeline.

A single prompt (1 LLM call) for draft generation:

  1. Analyse instruction + schema
  2. Extract evidence from text
  3. Reason about values
  4. Produce <Thinking>…</Thinking><Draft>campo: valor</Draft>

The draft is **semi-structured** — NOT JSON. Just ``campo: valor`` lines.
The decoder (Phase 2, constrained decoding) handles the final JSON formatting.
"""


# ── Single Draft Prompt (merged A+B) ─────────────────────

# [DEPRECATED] Versión anterior con 4 bloques (A, B, C, D).
# ~3,180 chars / ~1,060 tok. Se mantiene como referencia.
#
# # ── Single Draft Prompt (merged A+B) ────────────────────────────────────
#
# DRAFT_PROMPT = (
#     "=== TAREA ===\n"
#     "{instruction}\n"
#     "\n"
#     "=== TEXTO ===\n"
#     "{input_text}\n"
#     "\n"
#     "=== ESQUEMA OBJETIVO ===\n"
#     "{schema_json}\n"
#     "\n"
#     "=== INSTRUCCIONES ===\n"
#     "\n"
#     "A. Análisis inicial\n"
#     "\n"
#     "   Antes de extraer, analiza la instrucción y el esquema:\n"
#     "\n"
#     "   1. Descompón la instrucción en sus elementos clave:\n"
#     "      - ¿Qué entidades o conceptos principales pide identificar?\n"
#     "      - ¿Qué atributos de cada entidad?\n"
#     "      - ¿Hay relaciones entre entidades?\n"
#     "\n"
#     "   2. Mapea cada campo del esquema a un elemento de la instrucción:\n"
#     "\n"
#     "      a) Usa la \"description\" de cada campo para entender qué parte\n"
#     "         de la instrucción le corresponde.\n"
#     "\n"
#     "      b) Determina, en el contexto de la instrucción, qué tipo de\n"
#     "         información necesitas para responder ese campo:\n"
#     "         - ¿Es un nombre propio? → búscalo como entidad nombrada.\n"
#     "         - ¿Es una clasificación? → busca pistas categóricas.\n"
#     "         - ¿Es una fecha? → busca formatos de fecha o referencias\n"
#     "           temporales.\n"
#     "         - ¿Es una lista? → busca enumeraciones en el texto.\n"
#     "\n"
#     "      c) Si la descripción de un campo es muy compleja o abarca\n"
#     "         múltiples aspectos, descomponla en sub-preguntas más\n"
#     "         sencillas. Cada sub-pregunta te ayudará a identificar\n"
#     "         fragmentos de evidencia más concretos en el texto.\n"
#     "\n"
#     "   3. Anticipa el tipo de extracción por campo:\n"
#     "      - Descripción concreta (\"nombre del software\", \"fecha del\n"
#     "        contrato\") → evidencia LITERAL.\n"
#     "      - Campos enum → requieren mapear descripciones del texto a\n"
#     "        etiquetas predefinidas.\n"
#     "      - Booleanos y puntuaciones → suelen requerir INFERENCIA.\n"
#     "      - Arrays → suelen requerir evidencia AGREGADA.\n"
#     "\n"
#     "   4. Identifica la estructura esperada:\n"
#     "      - ¿Hay objetos anidados? ¿Sub-campos requeridos?\n"
#     "      - ¿Hay arrays de objetos?\n"
#     "      - Esto te prepara para generar el JSON sin errores de estructura.\n"
#     "\n"
#     "B. Extracción de evidencia y determinación de valores\n"
#     "\n"
#     "   Para cada campo del esquema, busca evidencia en el texto y\n"
#     "   determina el valor:\n"
#     "\n"
#     "   • Evidencia LITERAL — el valor aparece textualmente →\n"
#     "     cópialo exacto.\n"
#     "   • Evidencia DESCRIPTIVA — el texto describe el concepto sin\n"
#     "     usar la palabra exacta → mapea al valor del esquema que\n"
#     "     mejor corresponda semánticamente.\n"
#     "   • Evidencia INFERIDA — requiere combinar múltiples pistas\n"
#     "     textuales → razona en <Thinking> y documenta la decisión.\n"
#     "   • Evidencia AGREGADA — información dispersa en varios\n"
#     "     fragmentos (típico de arrays) → incluye TODOS los elementos\n"
#     "     mencionados en el texto. No omitas ninguno.\n"
#     "   • Sin evidencia:\n"
#     "     - Campo escalar (string, enum, number) → null\n"
#     "     - Booleano sin inferencia posible → null (no false)\n"
#     "     - Array sin elementos en el texto → []\n"
#     "\n"
#     "C. Reglas de formato para los VALORES\n"
#     "\n"
#     "   • Fechas → YYYY-MM-DD\n"
#     "   • Enums → valor EXACTAMENTE como está en el esquema\n"
#     "     (case-sensitive)\n"
#     "   • Números → punto decimal, sin separadores de miles\n"
#     "   • Booleanos → true / false (sin comillas, en minúscula)\n"
#     "   • null → escribir null (sin comillas)\n"
#     "   • Arrays vacíos → []\n"
#     "   • Arrays con elementos → separados por coma, ej:\n"
#     "     partes: Empresa A, Empresa B, Empresa C\n"
#     "   • Idioma: conserva el idioma original del texto. NO traduzcas\n"
#     "     los valores extraídos. Si el texto está en español, los\n"
#     "     valores deben quedar en español.\n"
#     "\n"
#     "D. Jerarquía de decisión\n"
#     "\n"
#     "   Al resolver conflictos entre fuentes, aplica este orden de\n"
#     "   precedencia (de mayor a menor):\n"
#     "\n"
#     "   1. Evidencia literal clara → usar directamente.\n"
#     "   2. Evidencia descriptiva → mapear al valor del esquema.\n"
#     "   3. Evidencia ambigua → razonar y documentar en <Thinking>.\n"
#     "   4. Sin evidencia → null / [] según Bloque B.\n"
#     "\n"
#     "=== FORMATO DE RESPUESTA ===\n"
#     "\n"
#     "<Thinking>\n"
#     "[Razonamiento campo por campo: explica qué evidencia encontraste,\n"
#     " qué decisión tomaste y por qué.]\n"
#     "</Thinking>\n"
#     "<Draft>\n"
#     "nombre_del_campo: valor\n"
#     "otro_campo: valor\n"
#     "...uno por línea, SOLO los campos y sus valores finales...\n"
#     "</Draft>\n"
#     "\n"
#     "=== EJEMPLO DE FORMATO ===\n"
#     "\n"
#     "<Thinking>\n"
#     "Campo contract_title: evidencia literal encontrada en línea 1.\n"
#     "Campo contract_type: el texto dice \"prestación de servicios\" → mapea a PRESTACIÓN_DE_SERVICIOS.\n"
#     "Campo effective_date: \"14 de abril de 2026\" → normalizar a 2026-04-14.\n"
#     "</Thinking>\n"
#     "<Draft>\n"
#     "contract_title: ACUERDO MARCO DE PRESTACIÓN DE SERVICIOS METAFÍSICOS\n"
#     "contract_type: PRESTACIÓN_DE_SERVICIOS\n"
#     "effective_date: 2026-04-14\n"
#     "parties: Mente S.A., Espíritu Ltda.\n"
#     "total_clauses: 12\n"
#     "has_nda_clause: true\n"
#     "monetary_amount: null\n"
#     "</Draft>\n"
# )
#
#
#

# ── Single Draft Prompt (merged A+B) ─────────────────────

# [DEPRECATED] Versión anterior reducida v1 (~1,551 chars / ~517 tok).
# Se mantiene como referencia.
#
# # ── Nueva versión reducida (~1,200 chars / ~400 tok) ──────────────
#
# DRAFT_PROMPT = (
#     "=== TAREA ===\n"
#     "{instruction}\n"
#     "\n"
#     "=== TEXTO ===\n"
#     "{input_text}\n"
#     "\n"
#     "=== ESQUEMA OBJETIVO ===\n"
#     "{schema_json}\n"
#     "\n"
#     "=== INSTRUCCIONES ===\n"
#     "\n"
#     "Para cada campo del esquema, busca evidencia en el texto y determina el valor:\n"
#     "\n"
#     "• LITERAL — el valor aparece textualmente → cópialo exacto.\n"
#     "• DESCRIPTIVA — el texto describe el concepto sin usar la palabra exacta\n"
#     "  → mapea semánticamente al valor del esquema (enum, booleano, etc.).\n"
#     "• INFERIDA — requiere combinar múltiples pistas textuales\n"
#     "  → razona en <Thinking> y documenta la decisión.\n"
#     "• AGREGADA — información dispersa en varios fragmentos (típico de arrays)\n"
#     "  → incluye TODOS los elementos mencionados. No omitas ninguno.\n"
#     "\n"
#     "Sin evidencia:\n"
#     "  - Campo escalar (string, enum, number) → null\n"
#     "  - Booleano sin inferencia posible → null (no false)\n"
#     "  - Array sin elementos → []\n"
#     "\n"
#     "Formato:\n"
#     "  - Fechas → YYYY-MM-DD\n"
#     "  - Enums → valor EXACTO del esquema (case-sensitive)\n"
#     "  - Números → punto decimal, sin separadores\n"
#     "  - Booleanos → true / false (minúscula, sin comillas)\n"
#     "  - null → sin comillas\n"
#     "  - Arrays vacíos → []\n"
#     "  - Idioma: conserva el original. NO traduzcas.\n"
#     "\n"
#     "Precedencia:\n"
#     "  1. Evidencia literal → usar directamente.\n"
#     "  2. Evidencia descriptiva → mapear al esquema.\n"
#     "  3. Evidencia inferida/ambigua → razonar en <Thinking>.\n"
#     "  4. Sin evidencia → null / [] según reglas.\n"
#     "\n"
#     "=== FORMATO DE RESPUESTA ===\n"
#     "\n"
#     "<Thinking>\n"
#     "[Razonamiento campo por campo: evidencia encontrada y decisión.]\n"
#     "</Thinking>\n"
#     "<Draft>\n"
#     "campo: valor\n"
#     "campo: valor\n"
#     "...uno por línea, SOLO los campos y sus valores...\n"
#     "</Draft>\n"
# )
#
#

# ── Single Draft Prompt (merged A+B) ─────────────────────

# [DEPRECATED] Versión refinada v2 (~1,077 chars / ~359 tok).
# Se mantiene como referencia.
#
# # ── Versión refinada v2 (~1,077 chars / ~359 tok) ─────────────────
#
# DRAFT_PROMPT = (
#     "=== TAREA ===\n"
#     "{instruction}\n"
#     "\n"
#     "=== TEXTO ===\n"
#     "{input_text}\n"
#     "\n"
#     "=== ESQUEMA OBJETIVO ===\n"
#     "{schema_json}\n"
#     "\n"
#     "=== INSTRUCCIONES ===\n"
#     "\n"
#     "Extrae cada campo del esquema buscando evidencia en el texto:\n"
#     "\n"
#     "\u2022 LITERAL \u2192 copia textual.\n"
#     "\u2022 DESCRIPTIVA \u2192 mapea sem\u00e1nticamente al esquema.\n"
#     "\u2022 INFERIDA \u2192 razona en <Thinking>.\n"
#     "\u2022 AGREGADA (arrays) \u2192 busca en TODO el texto. S\u00e9 exhaustivo.\n"
#     "\n"
#     "CR\u00cdTICO:\n"
#     "1. null = AUSENCIA en el texto. NO infieras, NO defaults. null, no \"null\".\n"
#     "2. Enums/strings \u2192 t\u00e9rmino EXACTO del esquema (case-sensitive). NO sin\u00f3nimos.\n"
#     "3. Arrays \u2192 SOLO elementos expl\u00edcitamente mencionados. Si 0 \u2192 [].\n"
#     "\n"
#     "Sin evidencia:\n"
#     "  Escalar \u2192 null  |  Booleano \u2192 null  |  Array \u2192 []\n"
#     "\n"
#     "Formato:\n"
#     "  Fechas \u2192 YYYY-MM-DD  |  N\u00fameros \u2192 decimal\n"
#     "  Booleanos \u2192 true/false  |  null \u2192 sin comillas\n"
#     "  Idioma original. NO traduzcas.\n"
#     "\n"
#     "=== FORMATO DE RESPUESTA ===\n"
#     "\n"
#     "<Thinking>\n"
#     "[Razonamiento campo por campo.]\n"
#     "</Thinking>\n"
#     "<Draft>\n"
#     "campo: valor\n"
#     "campo: valor\n"
#     "</Draft>\n"
# )
#
#

# ── Single Draft Prompt (merged A+B) ─────────────────────

# [DEPRECATED] Versión compacta v3 (~1,621 chars / ~540 tok).
# Causó desbordamiento de contexto. Se mantiene como referencia.
#
# # ── Versión compacta v3 (~1,621 chars / ~540 tok) ─────────────────
#
# DRAFT_PROMPT = (
#     "=== TAREA ===\n"
#     "{instruction}\n"
#     "\n"
#     "=== TEXTO ===\n"
#     "{input_text}\n"
#     "\n"
#     "=== ESQUEMA OBJETIVO ===\n"
#     "{schema_json}\n"
#     "\n"
#     "=== INSTRUCCIONES ===\n"
#     "\n"
#     "Extrae cada campo del esquema buscando evidencia en el texto:\n"
#     "\n"
#     "TIPOS DE EVIDENCIA:\n"
#     "\u2022 LITERAL \u2192 copia textual exacta.\n"
#     "\u2022 DESCRIPTIVA \u2192 el texto describe el concepto \u2192 mapea al esquema.\n"
#     "\u2022 INFERIDA \u2192 combina pistas del texto \u2192 razona en <Thinking>.\n"
#     "\u2022 AGREGADA (arrays) \u2192 re\u00fane informaci\u00f3n dispersa.\n"
#     "\n"
#     "CR\u00cdTICO:\n"
#     "\n"
#     "[1] VALORES EXACTOS:\n"
#     "  Strings del texto \u2192 palabras LITERALES. No parafrasees,\n"
#     "  no expandas siglas, no cambies formato.\n"
#     "  Enums del esquema \u2192 t\u00e9rmino EXACTO (case-sensitive).\n"
#     "  null = AUSENCIA. No infieras severity_level, unit, amount,\n"
#     "  fechas, nombres, etc. sin evidencia textual.\n"
#     "  Nunca uses \"\" ni \"null\". Booleanos sin evidencia \u2192 null.\n"
#     "\n"
#     "[2] INFERENCIA CONDICIONAL:\n"
#     "  Lee la description de cada campo en el esquema.\n"
#     "  Si dice \"Reasoned\", \"inferred\", \"based on\" \u2192 aplica INFERIDA.\n"
#     "  Si no menciona inferencia \u2192 solo LITERAL o DESCRIPTIVA.\n"
#     "\n"
#     "[3] ARRAYS:\n"
#     "  Etiquetas CORTAS (1-4 palabras). Una capacidad = una etiqueta.\n"
#     "  Listas >5: REVISA el texto completo. Cuenta mentalmente.\n"
#     "  No omitas ning\u00fan elemento mencionado.\n"
#     "  technique_sequence, steps, timeline: respeta el ORDEN del texto.\n"
#     "\n"
#     "Formato:\n"
#     "  Fechas \u2192 YYYY-MM-DD | N\u00fameros \u2192 decimal\n"
#     "  Booleanos \u2192 true/false | null \u2192 sin comillas\n"
#     "  Idioma original. NO traduzcas.\n"
#     "\n"
#     "=== FORMATO DE RESPUESTA ===\n"
#     "\n"
#     "<Thinking>\n"
#     "[Razonamiento campo por campo.]\n"
#     "</Thinking>\n"
#     "<Draft>\n"
#     "campo: valor\n"
#     "campo: valor\n"
#     "</Draft>\n"
# )
#
#

# ── Versión v3b (~1,163 chars / ~387 tok) ───────────────────

DRAFT_PROMPT = (
    "=== TAREA ===\n"
    "{instruction}\n"
    "\n"
    "=== TEXTO ===\n"
    "{input_text}\n"
    "\n"
    "=== ESQUEMA OBJETIVO ===\n"
    "{schema_json}\n"
    "\n"
    "=== INSTRUCCIONES ===\n"
    "\n"
    "Para CADA campo, busca evidencia en el texto y clasif\u00edcala:\n"
    "\n"
    "LITERAL \u2192 el valor aparece textualmente \u2192 c\u00f3pialo exacto.\n"
    "DESCRIPTIVA \u2192 el texto describe el concepto \u2192 mapea al esquema.\n"
    "INFERIDA \u2192 combina \u22652 pistas textuales \u2192 razona en <Thinking>.\n"
    "AGREGADA (arrays) \u2192 re\u00fane info dispersa en el texto.\n"
    "\n"
    "REGLAS:\n"
    "\n"
    "[INFERENCIA]\n"
    "  S\u00f3lo INFERIDA si la description del campo en el esquema\n"
    "  contiene: reasoned, inferred, based on, classify.\n"
    "  PROHIBIDO inferir: idioma, nacionalidad, fecha de publicaci\u00f3n,\n"
    "  ni ning\u00fan metadato externo que no est\u00e9 en el texto.\n"
    "  Inferir \"el texto es espa\u00f1ol porque habla de Espa\u00f1a\"\n"
    "  es un error. La inferencia solo combina EVIDENCIA TEXTUAL.\n"
    "\n"
    "[VALORES]\n"
    "  Strings \u2192 palabras EXACTAS del texto. No parafrasees.\n"
    "  Enums \u2192 t\u00e9rmino EXACTO del esquema (case-sensitive).\n"
    "  null = AUSENCIA de evidencia. No \"\" ni \"null\".\n"
    "  Arrays \u2192 TODOS los elementos mencionados. 0 \u2192 [].\n"
    "\n"
    "[FORMATO]\n"
    "  Para CADA campo, lee su description en el esquema:\n"
    "  la description especifica el formato esperado.\n"
    "  SIGUE la description.\n"
    "  NO uses reglas de formato fijas.\n"
    "\n"
    "=== FORMATO DE RESPUESTA ===\n"
    "\n"
    "<Thinking>[Razonamiento campo por campo.]</Thinking>\n"
    "<Draft>\n"
    "campo: valor\n"
    "campo: valor\n"
    "</Draft>\n"
)

# ── Constrained Decoding (Draft-as-Context) ────────────────────────────

# [DEPRECATED] Versión anterior — más verbosa, con role prompt e instrucciones
# de formato redundantes (el response_format=json_schema ya las garantiza).
# Se mantiene como referencia por si la versión reducida causa problemas.
#
# CD_DRAFT_CONTEXT = (
#     "=== TAREA ===\n"
#     "Eres un traductor. Tu única función es copiar los valores del borrador\n"
#     "dentro de la estructura del esquema JSON.\n"
#     "\n"
#     "No verificas, no infieres, no re-extraes. Solo traduces.\n"
#     "\n"
#     "=== BORRADOR ===\n"
#     "{draft_json}\n"
#     "\n"
#     "=== INSTRUCCIONES ===\n"
#     "1. Copia cada valor del borrador a su campo correspondiente en el esquema.\n"
#     "2. Si un valor necesita ajuste de formato para encajar en el esquema:\n"
#     "   - Fechas → YYYY-MM-DD\n"
#     "   - Enums → case-sensitive exacto\n"
#     "   - Números → punto decimal, sin separadores\n"
#     "   - Booleanos → true/false\n"
#     "3. Si un campo del esquema no está en el borrador → null (o [] si es array).\n"
#     "4. Si un campo del borrador no está en el esquema → ignóralo.\n"
#     "5. No añadas nada que no esté en el borrador. No infieras valores faltantes.\n"
#     "6. Conserva el idioma original de los valores. NO traduzcas.\n"
# )

CD_DRAFT_CONTEXT = (
    "Copia los valores del borrador al esquema JSON de salida.\n"
    "\n"
    "=== BORRADOR ===\n"
    "{draft_json}\n"
    "\n"
    "=== INSTRUCCIONES ===\n"
    "- Output ONLY valid JSON. No thinking, no explanation.\n"
    "- Start directly with {{.\n"
    "- Campos del esquema con valor en el borrador → copia el valor.\n"
    "- Campos del esquema SIN valor en el borrador → null (o [] si es array).\n"
    "- Campos del borrador que no están en el esquema → ignóralos.\n"
    "- Conserva el idioma original de los valores.\n"
)



# ── Hybrid CoT Prompt (visible CoT + direct JSON) ─────────────────────
# Almost identical to DRAFT_PROMPT v3b but outputs JSON instead of
# <Draft>campo: valor</Draft>. Used by HybridCoTAgent.

HYBRID_COT_PROMPT = (
    "=== TAREA ===\n"
    "{instruction}\n"
    "\n"
    "=== TEXTO ===\n"
    "{input_text}\n"
    "\n"
    "=== ESQUEMA OBJETIVO ===\n"
    "{schema_json}\n"
    "\n"
    "=== INSTRUCCIONES ===\n"
    "\n"
    "Para CADA campo, busca evidencia en el texto y clasif\u00edcala:\n"
    "\n"
    "LITERAL \u2192 el valor aparece textualmente \u2192 c\u00f3pialo exacto.\n"
    "DESCRIPTIVA \u2192 el texto describe el concepto \u2192 mapea al esquema.\n"
    "INFERIDA \u2192 combina \u22652 pistas textuales \u2192 razona en <Thinking>.\n"
    "AGREGADA (arrays) \u2192 re\u00fane info dispersa en el texto.\n"
    "\n"
    "REGLAS POR TIPO DE CAMPO:\n"
    "\n"
    "  Para cada campo, la description del esquema define\n"
    "  QU\u00c9 extraer y C\u00d3MO extraerlo. Sigue estrictamente\n"
    "  la description del campo.\n"
    "\n"
    "  [STRINGS con texto del art\u00edculo]\n"
    "  Extrae el fragmento EXACTO del texto original.\n"
    "  NO parafrasees. NO resumas. NO reescribas.\n"
    "  NO a\u00f1adas prefijos (#, -, *) ni sufijos.\n"
    "  Copia textualmente, caracter por caracter.\n"
    "\n"
    "  [ENUMS]\n"
    "  Busca los valores permitidos en la description\n"
    "  del campo. MAPEA el texto al valor EXACTO del\n"
    "  enum (case-sensitive). NO uses sin\u00f3nimos.\n"
    "  NO copies texto literal en un campo enum.\n"
    "\n"
    "  [NUM\u00c9RICOS]\n"
    "  El n\u00famero debe estar EXPL\u00cdCITAMENTE escrito\n"
    "  en el texto. Si no hay n\u00famero expl\u00edcito \u2192 null.\n"
    "  NO infieras, aproximes ni calcules.\n"
    "\n"
    "  [BOOLEANOS]\n"
    "  Solo true si el texto afirma expl\u00edcitamente\n"
    "  el hecho. Si no hay afirmaci\u00f3n expl\u00edcita \u2192 null.\n"
    "  NO infieras del contexto.\n"
    "\n"
    "  [ARRAYS]\n"
    "  Identifica TODOS los elementos mencionados\n"
    "  en el texto. Cu\u00e9ntalos mentalmente. Verifica\n"
    "  que no hayas omitido ninguno ni a\u00f1adido extras.\n"
    "  Si hay 0 elementos \u2192 [].\n"
    "\n"
    "[INFERENCIA PROHIBIDA]\n"
    "  No infieras: idioma, nacionalidad, fecha de\n"
    "  publicaci\u00f3n, ni ning\u00fan metadato externo.\n"
    "  La inferencia solo es v\u00e1lida si la description\n"
    "  del campo contiene: reasoned, inferred,\n"
    "  based on, classify.\n"
    "\n"
    "=== FORMATO DE RESPUESTA ===\n"
    "\n"
    "<Thinking>\n"
    "[Razonamiento campo por campo: explica qu\u00e9 evidencia encontraste,\n"
    " qu\u00e9 decisi\u00f3n tomaste y por qu\u00e9.]\n"
    "</Thinking>\n"
    "\n"
    "Paso 1: Razona en <Thinking> ... </Thinking>.\n"
    "Paso 2: Genera el JSON de salida v\u00e1lido.\n"
    "El JSON debe cumplir EXACTAMENTE con el esquema objetivo.\n"
    "NO incluyas nada despu\u00e9s del JSON.\n"
)


# ── Single-Call Prompt (CoT + direct JSON) ────────────────────────────

SINGLE_CALL_PROMPT = (
    "=== TAREA ===\n"
    "{instruction}\n"
    "\n"
    "=== TEXTO ===\n"
    "{input_text}\n"
    "\n"
    "=== ESQUEMA OBJETIVO ===\n"
    "{schema_json}\n"
    "\n"
    "=== INSTRUCCIONES ===\n"
    "\n"
    "Para CADA campo, busca evidencia en el texto y clasif\u00edcala:\n"
    "\n"
    "LITERAL \u2192 el valor aparece textualmente \u2192 c\u00f3pialo exacto.\n"
    "DESCRIPTIVA \u2192 el texto describe el concepto \u2192 mapea al esquema.\n"
    "INFERIDA \u2192 combina \u22652 pistas textuales.\n"
    "AGREGADA (arrays) \u2192 re\u00fane info dispersa.\n"
    "\n"
    "REGLAS:\n"
    "\n"
    "[INFERENCIA]\n"
    "  S\u00f3lo INFERIDA si la description del campo en el esquema\n"
    "  contiene: reasoned, inferred, based on, classify.\n"
    "  PROHIBIDO inferir: idioma, nacionalidad, fecha de publicaci\u00f3n,\n"
    "  ni ning\u00fan metadato externo que no est\u00e9 en el texto.\n"
    "  La inferencia solo combina EVIDENCIA TEXTUAL.\n"
    "\n"
    "[VALORES]\n"
    "  Strings \u2192 palabras EXACTAS del texto. No parafrasees.\n"
    "  Enums \u2192 t\u00e9rmino EXACTO del esquema (case-sensitive).\n"
    "  null = AUSENCIA de evidencia. No \"\" ni \"null\".\n"
    "  Arrays \u2192 TODOS los elementos mencionados. 0 \u2192 [].\n"
    "\n"
    "[FORMATO]\n"
    "  Para CADA campo, lee su description en el esquema:\n"
    "  la description especifica el formato esperado.\n"
    "  SIGUE la description. NO uses reglas de formato fijas.\n"
    "\n"
    "Antes de generar el JSON, piensa internamente paso a paso.\n"
    "Revisa CADA campo contra el texto, aplica las reglas de\n"
    "evidencia, y luego genera el JSON de salida.\n"
)


# ── Helpers ────────────────────────────────────────────────────────────

def parse_draft_response(response: str) -> str | None:
    """Extract the best available content from a model response.

    Priority:
      1. Content inside <Draft>...</Draft>  (semi-structured, campo: valor lines)
      2. Content inside <Thinking>...</Thinking>  (fallback if no Draft)
      3. ``None`` — nothing usable found.

    Returns a single string (never a tuple). The caller passes this directly
    to the decoder.
    """
    # 1. Try <Draft>...</Draft>
    d_start = response.find("<Draft>")
    d_end = response.find("</Draft>")
    if d_start != -1 and d_end != -1:
        draft = response[d_start + len("<Draft>"):d_end].strip()
        if draft:
            return draft

    # 2. Fallback: <Thinking>...</Thinking>
    t_start = response.find("<Thinking>")
    t_end = response.find("</Thinking>")
    if t_start != -1 and t_end != -1:
        thinking = response[t_start + len("<Thinking>"):t_end].strip()
        if thinking:
            return thinking

    # 3. Nothing usable
    return None

