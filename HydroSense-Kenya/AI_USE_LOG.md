# AI Use Log – HydroSense-Kenya
**ICS 2207 Scientific Computing | February – May 2026**

| Level | Date | Tool | Prompt Used | AI Output Summary | Accepted? | Modified? | Validation Method |
|---|---|---|---|---|---|---|---|
| L1 | 2026-03-05 | Claude | "Draft a 500-word scientific problem statement for smart irrigation in Kenya" | Problem statement with context | Partly | Added Kenya Vision 2030 references and local crop data | Cross-checked against project brief requirements |
| L2 | 2026-03-12 | Claude | "Show me how to benchmark a Python loop vs NumPy using time.perf_counter" | Timing template with N_REPS | Yes | Adapted units to microseconds, added speedup calculation | Ran 5 times; variance < 5% across runs |
| L3 | 2026-03-19 | Claude | "Generate code to plot convergence error on a log scale for root-finding" | Semilogy plot with single method | Partly | Extended to include all three methods; fixed axis labels | Verified root x=2 for f(x)=x²-4 matches analytical value |
| L5 | 2026-04-02 | Claude | "Write a function to generate 1000 rainfall scenarios using numpy random" | Used unbounded normal distribution | Partly | Changed to clipped-normal (non-negative); added seed param | np.all(result >= 0) = True; mean/std match inputs within 5% |
| L6 | 2026-04-10 | Claude | "Generate pytest test cases for bisection root-finding function" | Tests for x²-4 only | Partly | Added irrigation-domain test, tightened tolerances to 1e-5 | All 16 tests pass; verified manually against known solutions |

## Responsible AI Use Notes
- All AI outputs were read, understood, and tested before inclusion.
- No AI output was submitted without manual validation.
- AI was used to accelerate repetitive coding and documentation tasks only.
- Scientific reasoning, interpretation, and domain knowledge were supplied by the student team.
