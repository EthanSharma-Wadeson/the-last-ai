# The Last AI — Open Morning Guide

A short talk you can give to **Year 5 / Year 6** (about 10 years old).

**Time:** ~5–8 minutes talk + a short demo if you have a laptop.

**Tone:** curious, honest, a bit magical — but clear that this is **a computer experiment**, not a real living creature.

---

## One-sentence pitch

> We built a tiny computer world full of little robot minds. Then we remove them one by one — and watch what the last one still “remembers.”

---

## What to say (kid-friendly script)

### 1. Hook (30 seconds)

Imagine a computer game with lots of little dots moving around.

They’re not people. They’re not pets. They’re **agents** — tiny robot programs that can:

- look around  
- walk  
- find “food”  
- bump into friends  
- send tiny messages like `food here` or `you where`  
- **remember** who they met  

That’s **The Last AI**.

### 2. The big question (1 minute)

Here’s the puzzle we care about:

> If your friends slowly leave a place…  
> what stays inside your head about them?

In school, you might still remember a friend who moved away.

Our agents aren’t human. They don’t feel sad the way you do.  
But their **computer memory** can still keep a picture of someone who isn’t there anymore.

We ask:

**When almost everyone disappears from the world… what does the last agent still have stored inside?**

### 3. How it works (2 minutes) — use your hands

Use three easy words:

1. **World** — a grid map (like graph paper) with walls and food.  
2. **Agents** — dots that move and learn.  
3. **Memory** — notes the computer keeps: “I saw agent_001 near here.”

What they can learn (keep it simple):

| Everyday idea | What the computer actually does |
|---------------|----------------------------------|
| “I know that friend” | Stores numbers about trust / how often they met |
| “I expect them here” | Predicts where they might be |
| “They’re gone?” | Looks for them, gets a mismatch, updates scores |
| “Talking” | Sends short symbol words, not real English chat |

**Important honesty line (say this clearly):**

> When the screen says things like “social loss,” that is a **score**, not a real feeling.  
> It’s like a thermometer for “how surprised / how much searching the program is doing” — not crying.

### 4. The cool demo moments (1–2 minutes)

If you can run the spectator on a laptop **(recommended for open morning)**:

```bash
source .venv/bin/activate
last-ai spectate --demo open --seed 0
```

This version **loops forever**: friends meet → one leaves → searching/scores → world shrinks → last agent → restart.

Leave it running on a big screen. Kids can walk up any time and still catch the story.

Other demos (single run, no loop):

```bash
last-ai spectate --demo loss --seed 0 --speed 6
last-ai spectate --demo collapse --seed 0 --agents 24 --speed 8
```

**What kids should watch for:**

1. Two agents hang out together (bonding).  
2. One vanishes.  
3. The other may walk toward the empty spot (searching memory).  
4. Numbers on the side change (prediction error / social loss).  
5. Tiny messages may appear (computer symbols, not emotions).

If you can’t demo live: show a screenshot or describe it with the board.

### 5. Why this matters (1 minute)

We’re practising **careful science about AI**:

- Computers can store patterns about the past.  
- Patterns can keep affecting behaviour after something is gone.  
- That does **not** mean the computer is alive, conscious, or sad.  
- Good AI research means measuring what happens — not making scary claims.

Close with:

> The Last AI is a story about **memory in machines** —  
> what stays when the world gets quiet.

---

## Super-short version (if you only have 2 minutes)

1. Tiny robot minds live in a computer world.  
2. They learn and remember each other.  
3. We remove them until one is left.  
4. We look at what memory is still inside.  
5. Scores ≠ feelings. Science, not sci-fi horror.

---

## Words that help (and words to avoid)

**Say:**

- agent, computer program, memory, prediction, score, experiment, simulation  
- “behaves as if…” / “looks like searching”  
- “computational response”

**Avoid claiming:**

- “It’s sad / lonely / grieving” as a fact  
- “It’s conscious / alive / has a soul”  
- “We’re hurting AIs”

**OK as analogy (then correct):**

> “It *looks a bit like* missing someone — but really it’s numbers and searching.”

---

## Questions kids often ask

**Q: Are they alive?**  
A: No. They’re programs, like a very clever calculator that can move on a map.

**Q: Do they feel pain?**  
A: No. We don’t give them feelings. We measure memory and behaviour.

**Q: Why make them disappear?**  
A: To study what memory does when the world changes — like a science experiment with a clear question.

**Q: Can it take over the world?**  
A: No. It only runs on this computer, in a tiny grid. It’s a research toy, not a monster.

**Q: Why do they say weird things like “no gone”?**  
A: They only know a tiny word list. It’s symbols from their state — not a poem, not a chat bot.

**Q: What’s the point?**  
A: To understand how AI memory works when things change — carefully and honestly.

---

## Hands-on activity (optional, 3 minutes)

No laptop needed:

1. Ask 5 volunteers to be “agents.”  
2. Give each a sticky note: “I remember ___.”  
3. Have them stand near each other and write a friend’s name.  
4. Ask 4 to sit down (“disappear”).  
5. Ask the last one: “What’s still on your sticky note?”  

Then say:

> That’s like our project — the world got smaller, but a **record** can remain.

---

## If a teacher / parent asks something harder

- Scientific rule: **observed computer response ≠ subjective emotion**.  
- Full docs: `docs/LOSS.md`, `docs/SPECTATOR.md`, `docs/ETHICS.md`.  
- Code: public MIT project — https://github.com/EthanSharma-Wadeson/the-last-ai  

---

## Checklist before the morning

- [ ] Laptop charged  
- [ ] `.venv` works / `last-ai spectate --demo open` opens in browser and **loops**  
- [ ] One honesty sentence ready about scores ≠ feelings  
- [ ] One sticky-note activity ready as backup if Wi‑Fi fails  

Good luck — keep it curious, keep it honest, and let the kids ask wild questions.
