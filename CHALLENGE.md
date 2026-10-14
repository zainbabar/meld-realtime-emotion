# ML Challenge

## Real-Time Multimodal ML Challenge

---

## Context

We are exploring emotion-aware interactions for a character robot. Tone, expression, and timing can change how words should be understood. Build a small prototype that turns multimodal signals into real-time output.

Use MELD (Multimodal EmotionLines Dataset) as the shared foundation. It provides conversational text, audio, video, and emotion labels.

---

## Assignment

- Choose one primary track: **Text and Audio** or **Text and Vision**
- Combine utterance text with your chosen modality.
- Produce structured state or tags containing a MELD emotion category, plus short response text grounded in the multimodal input.
- Accept input over time and return results suitable for interaction.
- Deliver one coherent end-to-end prototype.

---

## Scope and freedom

- Define “real-time” for your intended interaction and explain your assumptions.
- Choose the problem framing, models, architecture, interfaces, and tools.
- Prefer a focused, understandable solution over a large unfinished one.

---

## Optional standout extension

- After completing the core task, optionally combine vision, voice, and text in one real-time system producing structured state or tags and streaming text.
- Show an interaction where all three inputs work together meaningfully.
- Optionally apply reinforcement learning to one component and demonstrate measured value.

---

## Demonstration and handoff

- Submit a runnable repository with clear guidance.
- Demonstrate one input travelling through the system to both outputs.
- Explain the decisions, evidence, trade-offs, and limitations.

---

## Constraints

- The complete local inference path may contain at most **6 billion total learned parameters**, counting every required encoder, backbone, adapter, projector, head, and auxiliary generator. Use total—not active, effective, or trainable—parameters.
- Report the hardware and observed resource requirements.
- Pretrained models, AI tools, and open-source tools are allowed. Remote inference does not satisfy the core task.
- Identify important external or generated components and own their results.
- State what you completed and what you intentionally left out.

---

## Timebox

Spend no more than two to three days. A small, correct, well-explained prototype is preferred over additional incomplete experiments.

---

## Use of AI tools

AI-assisted development tools are allowed. You remain responsible for the code, experiments, results, and technical decisions, and should be able to explain and modify the work.