# TODO — black-hole-sim-roadmap implementation

- [ ] Phase 0: Make “Hello Window” visually non-empty
  - [ ] Add a simple triangle render (basic shader + VBO/VAO) in `src/bhsim/app.py`
  - [ ] Verify: window shows non-black content; ESC/Q closes cleanly; FPS title updates
  - [ ] Verify config wiring: width/height/vsync/fullscreen apply without crash

- [ ] Phase 0: Verify & align project structure
  - [ ] Ensure `resources/` directory exists (already required by moderngl-window)
  - [ ] Ensure logging init happens exactly once and doesn’t duplicate handlers

- [ ] Phase 1: Math & physics foundations (CPU-only)
  - [ ] Confirm/complete `physics/units.py`, `physics/schwarzschild.py`, `physics/integrators.py`
  - [ ] Add/verify `utils/math_utils.py` coverage
  - [ ] Write any missing tests to fully cover the Phase 1 expectations

- [ ] Phase 2+: Implement layered architecture and proceed phase-by-phase per `black-hole-sim-roadmap.md`
  - [ ] Follow data flow: Input → Simulation Core → Uniform packing → gl shaders/pipeline
  - [ ] Enforce import rule: physics never imports gl_core directly
