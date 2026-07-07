# Prompt-Cache Optimization & Demand Paging in Hermes Agent

LLMs exposed via high-performance router gateways like OmniRoute or direct APIs (Anthropic, Gemini, OpenAI) heavily utilize **Prompt Caching** to speed up inference and drastically lower per-token costs. A single invalidating byte in the beginning of a conversation prompts a slow and expensive "prefill" phase.

To achieve maximum performance and cost efficiency, Hermes Agent employs the **"Static Prefix, Dynamic Suffix"** architecture paired with **"Demand Paging"** for skill metadata and guidelines.

---

## 1. The Core Principle: Static Prefix, Dynamic Suffix

Prompt caches are built sequentially. A cache hits when the prefix of the current prompt perfectly matches a previously cached prefix (usually exact byte-for-byte matching with a specific block granularity, e.g., 1024 tokens).

- **Static Prefix (Cached)**: System prompts, persona definitions, Core System rules, and loaded SKILL definitions (`SKILL.md`).
- **Dynamic Suffix (Mutable)**: User query, current timestamp, active process outputs, server resource logs, and interactive chat history.

### The Pollution Anti-Pattern
If a skill (`SKILL.md`) contains highly dynamic state (e.g., `Last checked: 2026-05-29 14:02`, `Active processes: 54`, `Available memory: 8.2GB`), this dynamic metadata sits high up in the system prompt. Every time an agent updates or reads this skill:
1. The byte-level change invalidates the cache for that turn.
2. The entire system prompt must be re-parsed (prefilled).
3. Prefill costs skyrocket and response latency increases 3-5x.

### Correct Implementation
Keep `SKILL.md` files **100% static**. Put all volatile, temporal, or user-specific facts into on-demand references or let the agent fetch them dynamically at runtime via terminal/file tools.

---

## 2. Demand Paging (Progressive Disclosure)

A skill should not be an omnibus technical manual. Loading a 20KB markdown file containing 15 different disaster recovery playbooks on every turn is extremely wasteful. 

Instead, implement **Demand Paging**:
1. **Core SKILL.md (The Index / Triggers)**: Contains the description, basic workflow (80/20 rule), common command anchors, and main pitfalls. (Ideally < 3-5 KB).
2. **References Subdirectory (`references/`)**: Contains detailed troubleshooting manuals, heavy config templates, edge-case logs, and historical deep dives.

### Example Directory Layout
```text
skills/devops/kuzhomesrv-storage/
├── SKILL.md                          # Main lightweight playbook (3KB)
└── references/
    ├── rclone-bisync-recovery.md      # Detailed recovery from deadlock (6KB)
    └── storage-layout-cheatsheet.md  # Raw disk/partition map (4KB)
```

Within the `SKILL.md`, guide the model on when to load these files:
> "If an rclone bisync lock occurs, do not attempt to guess the flags. IMMEDIATELY load and follow the instructions in `references/rclone-bisync-recovery.md`."

---

## 3. Designing Cache-Friendly Prompts (The 5 Golden Rules)

### Rule 1: No Dynamic Placeholders in System-Space
Never inject dynamic macros (like `{{CURRENT_TIME}}` or `{{SERVER_STATUS}}`) in the middle of static templates, unless they are positioned at the absolute end of the system prompt (immediately before the user prompt).

### Rule 2: Strict Isolation of Logs
When appending log clips or terminal stdout to chat context to analyze an issue, sanitize them to strip dynamic timestamps if they are repeated across turns, or truncate them tightly. Keep them in user replies, not system/agent definitions.

### Rule 3: Single "In_Progress" Mindset
In multi-turn execution, keep the task structure static inside the shell/terminal and only pull task updates when needed (using `todo_ide`).

### Rule 4: Structural Consistency
Stick to the same order of loaded skills. Hermes maintains skill order based on the loading sequence to increase prefix match probability.

### Rule 5: Keep It Declarative
When recording facts via `memory_ide`, write them as static declarative statements (`User prefers Obsidian WebDAV for note-taking`) rather than temporal logs (`On 2026-05-29 user selected WebDAV`).
