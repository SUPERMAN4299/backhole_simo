# Black Hole Visualization Engine — Master Roadmap

**Role of this document:** This is the architecture and roadmap for a long-term project. No source code is included yet, per your instructions. When you're ready, say **"Next Phase"** and we'll go through it: theory → design → implementation → testing → refactor.

---

## 1. High-Level Architecture

Think of the engine as four decoupled layers. This separation is the single most important architectural decision in the whole project — it's what keeps a raytraced-relativity renderer from turning into unmaintainable spaghetti by Phase 6.

```
┌──────────────────────────────────────────────────────────────┐
│                        Application Layer                      │
│   (main.py, App class, window/context lifecycle, game loop)   │
└───────────────┬─────────────────────────────┬─────────────────┘
                │                             │
     ┌──────────▼──────────┐       ┌──────────▼──────────┐
     │     UI Layer         │       │     Input Layer      │
     │  (imgui panels,      │       │  (camera controller, │
     │   config editing,    │       │   keybindings)        │
     │   debug overlays)    │       │                        │
     └──────────┬──────────┘       └──────────┬──────────────┘
                │                             │
     ┌──────────▼─────────────────────────────▼──────────────┐
     │                    Simulation Core                     │
     │  physics/  (metric, geodesics, integrators, units)      │
     │  scene/    (black hole params, disk params, camera)     │
     │  config/   (dataclasses, presets, serialization)        │
     └──────────┬───────────────────────────────────────────┘
                │
     ┌──────────▼───────────────────────────────────────────┐
     │                    Rendering Core                      │
     │  gl/        (context, buffers, VAO/VBO wrappers)         │
     │  shaders/   (GLSL: raymarch, lensing, disk, post-fx)     │
     │  pipeline/  (framebuffers, HDR, bloom, tonemap passes)   │
     │  assets/    (LUTs, noise textures, skyboxes)              │
     └──────────────────────────────────────────────────────┘
```

**Key principle:** physics never imports from `gl/`, and `gl/` never imports from `physics/` directly — they only meet inside `pipeline/`, which is responsible for translating simulation state into uniforms/buffers the GPU understands. This means you can unit-test all relativity math with plain NumPy/pytest, with zero OpenGL context required.

### 1.1 Data Flow (per frame)

```
Input → Camera update → Scene state (CPU, NumPy/GLM)
      → Uniform buffer packing (CPU→GPU)
      → GPU: ray generation → geodesic ray-march (fragment shader)
      → radiance accumulation (disk emission, lensed background, redshift/beaming)
      → HDR framebuffer
      → Bloom passes (downsample/blur/upsample)
      → Tonemap + gamma → LDR framebuffer
      → ImGui overlay
      → Present
```

### 1.2 Why ray marching in the fragment shader (not CPU ray tracing)

Photon geodesics near a black hole are curved paths, not straight lines, so we can't use a traditional rasterizer or a simple analytic ray-sphere intersection. The standard technique (used in real GR visualizers like the *Interstellar* black hole render, and various shader-toy implementations) is: for every pixel, integrate the photon's path backwards from the camera through curved spacetime using a numerical integrator (RK4), entirely on the GPU, in parallel, per pixel. We'll build a CPU version first (slow, correct, easy to debug) and port it to GLSL once verified — this is a deliberate phase in the roadmap below.

---

## 2. Folder Structure (target, end-state)

You won't create all of this on day one — it grows phase by phase. This is the target shape so every phase knows where its files belong.

```
blackhole-sim/
├── pyproject.toml
├── README.md
├── CHANGELOG.md
├── .gitignore
├── config/
│   ├── default.yaml
│   └── presets/
│       ├── sagittarius_a.yaml
│       └── stellar_mass.yaml
├── src/
│   └── bhsim/
│       ├── __init__.py
│       ├── app.py                  # Application entrypoint / main loop
│       ├── constants.py            # Physical constants, unit system
│       ├── logging_config.py
│       │
│       ├── physics/
│       │   ├── __init__.py
│       │   ├── units.py            # geometrized units, conversions
│       │   ├── schwarzschild.py    # metric, r_s, photon sphere, ISCO
│       │   ├── kerr.py             # (Phase 8+)
│       │   ├── geodesics.py        # null geodesic equations
│       │   ├── integrators.py      # RK4, RKF45, adaptive step
│       │   └── redshift.py         # gravitational + Doppler shift
│       │
│       ├── scene/
│       │   ├── __init__.py
│       │   ├── camera.py           # free camera + cinematic camera
│       │   ├── black_hole.py       # BlackHole dataclass (mass, spin)
│       │   ├── accretion_disk.py   # disk model parameters
│       │   └── scene_state.py      # aggregates full frame state
│       │
│       ├── gl_core/
│       │   ├── __init__.py
│       │   ├── context.py          # ModernGL context setup
│       │   ├── buffers.py          # VBO/VAO/UBO wrappers
│       │   ├── framebuffer.py      # FBO + HDR render targets
│       │   ├── shader_program.py   # shader loading/hot-reload
│       │   └── texture.py          # texture/LUT loading
│       │
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── raymarch_pass.py
│       │   ├── bloom_pass.py
│       │   ├── tonemap_pass.py
│       │   └── uniform_packing.py  # CPU state -> GPU uniforms
│       │
│       ├── ui/
│       │   ├── __init__.py
│       │   ├── imgui_layer.py
│       │   ├── panels/
│       │   │   ├── physics_panel.py
│       │   │   ├── camera_panel.py
│       │   │   └── render_panel.py
│       │   └── overlays/
│       │       └── debug_overlay.py
│       │
│       └── utils/
│           ├── __init__.py
│           ├── profiling.py
│           └── math_utils.py
│
├── shaders/
│   ├── common/
│   │   ├── constants.glsl
│   │   └── noise.glsl
│   ├── raymarch/
│   │   ├── geodesic.frag
│   │   └── fullscreen.vert
│   ├── postfx/
│   │   ├── bright_pass.frag
│   │   ├── blur.frag
│   │   └── tonemap.frag
│   └── disk/
│       └── disk_shading.glsl
│
├── assets/
│   ├── skyboxes/
│   ├── luts/
│   └── noise_textures/
│
├── tests/
│   ├── physics/
│   │   ├── test_schwarzschild.py
│   │   ├── test_geodesics.py
│   │   └── test_integrators.py
│   └── gl_core/
│       └── test_buffers.py
│
├── scripts/
│   ├── plot_geodesics.py           # matplotlib debug tool (CPU-only)
│   └── benchmark_integrator.py
│
└── docs/
    ├── architecture.md
    ├── physics_notes.md
    └── phase_journal/               # your dev log per phase
```

---

## 3. Full Development Roadmap

The project is organized into **4 Arcs**, containing **12 Phases**. Arcs group phases by what kind of engine you're building at that point.

| Arc | Phases | You end up with |
|---|---|---|
| **A — Foundations** | 0–2 | A working window with a triangle, a math/physics testing harness, and your architecture skeleton |
| **B — Flat-Space Renderer** | 3–5 | A textured, lit, camera-controllable 3D scene with HDR + bloom (no gravity yet) |
| **C — Curved Spacetime** | 6–9 | Real Schwarzschild lensing, event horizon, photon sphere, accretion disk, redshift/beaming |
| **D — Advanced** | 10–12 | Kerr (spinning) black holes, cinematic camera system, polish/portfolio packaging |

Each phase below follows the structure you requested: Goal, Concepts, Files, Physics, Graphics, Difficulty, Time, Result, Milestone, Mistakes, Testing, Refactoring, Performance.

---

### **Phase 0 — Environment, Tooling & Project Skeleton**

**Goal:** A clean, professional Python project scaffold with no rendering yet — just proof that your toolchain works.

**Concepts to Learn:**
- Python packaging (`pyproject.toml`, `src/` layout vs flat layout)
- Virtual environments, dependency pinning
- Git workflow: branches, conventional commits, `.gitignore` for Python/OpenGL projects
- Logging vs `print` debugging
- Config-driven design (why hardcoded constants are a trap)

**Files to Create:**
`pyproject.toml`, `src/bhsim/__init__.py`, `src/bhsim/app.py` (empty window only), `src/bhsim/logging_config.py`, `config/default.yaml`, `.gitignore`, `README.md`

**Physics:** None yet.

**Graphics:** Open a ModernGL/moderngl-window context, clear the screen to a color, confirm VSync and frame timing work.

**Estimated Difficulty:** ★☆☆☆☆ (Low)
**Estimated Time:** 2–4 hours

**Expected Result:** `python -m bhsim.app` opens a resizable window that clears to a dark color at a stable frame rate, with a logger printing startup info.

**Milestone:** ✅ "Hello Window" — window opens, closes cleanly, logs FPS.

**Common Mistakes:**
- Putting business logic directly in the window/app class (breaks testability later).
- Not pinning dependency versions (ModernGL/GLFW backends can behave differently across versions).
- Skipping `.gitignore` and committing `__pycache__`/build artifacts.

**Testing Strategy:** No unit tests yet — but set up `pytest` and write one trivial test (e.g., config loads correctly) so CI/test tooling exists from day one.

**Refactoring Tasks:** None yet (nothing to refactor).

**Performance Notes:** Confirm VSync behavior; measure baseline empty-frame cost so later phases have a reference point.

---

### **Phase 1 — Math & Physics Foundations (CPU-only, no graphics)**

**Goal:** Build and test the core mathematical/physics utilities in pure Python/NumPy, fully decoupled from rendering. This is the phase where "physics never imports from gl_core" gets proven true.

**Concepts to Learn:**
- Vectors, dot/cross products, normalization
- Coordinate systems: Cartesian vs spherical vs Boyer-Lindquist (preview)
- Units: geometrized units (G = c = 1) and why GR uses them
- Schwarzschild radius, escape velocity derivation
- Basic differential equations & why we need numerical integration
- Runge-Kutta 4th order (RK4) method, step size, error accumulation

**Files to Create:**
`physics/units.py`, `physics/schwarzschild.py`, `physics/integrators.py`, `utils/math_utils.py`, `tests/physics/test_schwarzschild.py`, `tests/physics/test_integrators.py`, `scripts/plot_geodesics.py` (matplotlib, CPU sanity check)

**Physics:**
- Implement $r_s = \frac{2GM}{c^2}$, escape velocity $v_{esc} = \sqrt{2GM/r}$
- Implement the Schwarzschild metric coefficients
- Implement a generic RK4 integrator (state-vector agnostic, so it's reusable for geodesics later)
- Validate against a known simple ODE (e.g., simple harmonic oscillator) before trusting it on GR equations

**Graphics:** None — this phase is intentionally graphics-free. (You may use `matplotlib` purely as a debugging/plotting tool, not as part of the engine.)

**Estimated Difficulty:** ★★★☆☆ (Medium — first real physics/math phase)
**Estimated Time:** 6–10 hours (more if GR math is new to you)

**Expected Result:** A CPU script that computes and plots straight-line light paths bending slightly near a test mass, using RK4, validated numerically against analytic approximations (e.g., weak-field deflection angle $\delta\phi \approx 4GM/(c^2 b)$).

**Milestone:** ✅ "Physics Kernel Verified" — integrator passes unit tests, deflection angle matches theory within tolerance for large impact parameters.

**Common Mistakes:**
- Mixing unit systems (SI vs geometrized) inconsistently — the #1 source of "my black hole physics doesn't look right" bugs.
- Using Euler integration instead of RK4 (visibly wrong orbits/trajectories).
- Fixed step size too large near the photon sphere, causing integration blow-up.

**Testing Strategy:** `pytest` unit tests: `r_s` matches known values (e.g., Sun, Sagittarius A*), RK4 reproduces analytic solutions for simple test ODEs, weak-field light bending matches the classic GR approximation within a defined tolerance.

**Refactoring Tasks:** Extract integrator into a generic, metric-agnostic function so Phase 6 (real geodesics) and Phase 10 (Kerr) can reuse it without rewriting.

**Performance Notes:** Not a concern yet — this is CPU, non-realtime, correctness-first code. Note runtime cost of RK4 per photon here; you'll compare it to the GPU version's performance in Phase 6.

---

### **Phase 2 — Application Architecture & Config System**

**Goal:** Turn the Phase 0 skeleton into the real layered architecture (App / UI / Input / Simulation / Rendering) with a proper config system, before any complex rendering exists.

**Concepts to Learn:**
- Dataclasses for structured config (`BlackHoleParams`, `CameraParams`, `RenderSettings`)
- YAML serialization/deserialization
- Dependency injection basics (passing config down rather than using globals)
- Game-loop architecture: fixed vs variable timestep
- SOLID principles as applied to a real-time app (single-responsibility for each manager class)

**Files to Create:**
`scene/black_hole.py`, `scene/camera.py`, `scene/scene_state.py`, `config/default.yaml`, `config/presets/*.yaml`, `app.py` (full loop: update → render → present)

**Physics:** Wire `BlackHole` dataclass to the `physics/schwarzschild.py` functions from Phase 1 (e.g., `black_hole.schwarzschild_radius` computed property).

**Graphics:** Still minimal — clear screen, but now driven by `SceneState` rather than hardcoded values, proving the data flow works end-to-end.

**Estimated Difficulty:** ★★☆☆☆ (Low-Medium)
**Estimated Time:** 4–6 hours

**Expected Result:** Editing `config/default.yaml` (e.g., changing black hole mass) visibly changes logged/derived values at runtime without touching code.

**Milestone:** ✅ "Config-Driven Skeleton" — full layered architecture in place, physics and rendering fully decoupled and independently testable.

**Common Mistakes:**
- Letting the `App` class directly mutate rendering AND physics state (God Object anti-pattern).
- Hardcoding file paths instead of using a config-relative asset resolver.

**Testing Strategy:** Unit tests for config loading/validation (bad YAML should fail loudly, not silently).

**Refactoring Tasks:** Revisit Phase 0's `app.py` and split responsibilities into `App`, `InputController`, `SceneState` cleanly.

**Performance Notes:** None significant yet; this is a structural phase.

---

### **Phase 3 — Modern OpenGL Fundamentals (Flat 3D Scene)**

**Goal:** Learn core OpenGL concepts by rendering a real (non-relativistic) 3D scene: a sphere (stand-in for the black hole silhouette) and a flat ring (stand-in for the disk), with a free-fly camera.

**Concepts to Learn:**
- VBOs, VAOs, index buffers
- Vertex vs fragment shaders, the graphics pipeline
- Uniforms, uniform buffer objects (UBOs)
- Model/View/Projection matrices
- Perspective projection derivation
- Basic Phong/Blinn-Phong lighting (as a stepping stone, even though the final disk will use emissive-only shading)

**Files to Create:**
`gl_core/context.py`, `gl_core/buffers.py`, `gl_core/shader_program.py`, `shaders/raymarch/fullscreen.vert` (placeholder), a basic `shaders/debug/mesh.vert` + `.frag`, `scene/camera.py` (free-fly controls implemented for real now)

**Physics:** None directly — but this is where `black_hole.schwarzschild_radius` first gets used to size the rendered sphere, connecting physics to visuals for the first time.

**Graphics:** Full mesh rendering pipeline: generate sphere/ring geometry, upload to GPU, write MVP matrices, implement a free-orbit camera (mouse-drag rotate, scroll zoom, WASD fly).

**Estimated Difficulty:** ★★★☆☆ (Medium)
**Estimated Time:** 10–15 hours (this is the "learn real OpenGL" phase — budget generously if new to you)

**Expected Result:** A navigable 3D scene with a shaded sphere and ring, correct perspective, smooth camera controls.

**Milestone:** ✅ "First Real 3D Scene" — camera math and rendering pipeline both verified visually and numerically (e.g., unit-test the projection matrix construction).

**Common Mistakes:**
- Row-major vs column-major matrix confusion (GLM/OpenGL is column-major) — classic source of "everything is warped" bugs.
- Not normalizing normals after scaling (breaks lighting).
- Forgetting depth testing, causing draw-order artifacts.

**Testing Strategy:** Unit tests for matrix construction (compare PyGLM output against hand-computed expected matrices for known inputs). Visual test checklist for camera controls (documented manually, since visuals aren't easily unit-tested).

**Refactoring Tasks:** Extract camera math into a reusable, GL-independent `Camera` class (already GL-independent from Phase 2, now proven correct against real rendering).

**Performance Notes:** Establish an FPS overlay now — you'll need it as a baseline for every phase after this, especially once ray marching is introduced.

---

### **Phase 4 — Framebuffers, HDR & Textures**

**Goal:** Move from direct-to-screen rendering to an HDR render pipeline, and learn texture/framebuffer management before post-processing is layered on.

**Concepts to Learn:**
- Framebuffer objects (FBOs), render targets, color attachments
- HDR color (floating-point framebuffers) vs LDR
- Texture formats, mipmaps, filtering, wrapping modes
- Gamma correction and why it matters
- Basic tonemapping (Reinhard, ACES-approx) — teach the math, not just plug-and-play

**Files to Create:**
`gl_core/framebuffer.py`, `gl_core/texture.py`, `pipeline/tonemap_pass.py`, `shaders/postfx/tonemap.frag`

**Physics:** None — pure graphics infrastructure phase.

**Graphics:** Render the Phase 3 scene into an HDR floating-point FBO, then blit/tonemap it to the screen via a fullscreen-quad post pass. This full-screen-quad pattern is exactly what the ray-marching shader in Phase 6 will use, so it's deliberately introduced here first, in isolation.

**Estimated Difficulty:** ★★★☆☆ (Medium)
**Estimated Time:** 6–10 hours

**Expected Result:** Same visual scene as Phase 3, but now rendered through an HDR pipeline with correct gamma, indistinguishable to the eye but structurally very different under the hood — this is intentional: you're validating plumbing, not changing visuals yet.

**Milestone:** ✅ "HDR Pipeline Online" — can confirm HDR values >1.0 survive the pipeline and tonemap correctly (e.g., test with an intentionally overbright light source).

**Common Mistakes:**
- Applying gamma correction twice (once in shader, once via sRGB framebuffer format) — washed-out or overly dark images.
- Forgetting to resize FBOs on window resize.

**Testing Strategy:** Visual regression checklist; a debug uniform to force known HDR test values through the pipeline and confirm expected tonemapped output.

**Refactoring Tasks:** Generalize `framebuffer.py` to support multiple color attachments (needed for bloom in Phase 5).

**Performance Notes:** Introduce GPU timer queries or at least CPU-side pass timing; framebuffer resizing/reallocation costs matter.

---

### **Phase 5 — Post-Processing: Bloom & Look Development**

**Goal:** Implement a proper bloom pipeline (bright-pass → blur chain → composite) and establish the visual "look" of the renderer before physics complexity is added.

**Concepts to Learn:**
- Bright-pass thresholding
- Gaussian blur (separable horizontal/vertical passes), and why separable blur is a performance technique, not just convenience
- Multi-pass downsample/upsample bloom (mip-chain bloom, like modern game engines)
- Additive blending/composition

**Files to Create:**
`pipeline/bloom_pass.py`, `shaders/postfx/bright_pass.frag`, `shaders/postfx/blur.frag`

**Physics:** None.

**Graphics:** Full HDR → bloom → tonemap chain. This is also a good phase to introduce a debug UI panel (first real imgui_bundle use) to tune bloom threshold/intensity live.

**Estimated Difficulty:** ★★★☆☆ (Medium)
**Estimated Time:** 6–8 hours

**Expected Result:** Bright elements in the scene (e.g., a test emissive sphere standing in for the future accretion disk) visibly glow, with tunable threshold/intensity via ImGui sliders.

**Milestone:** ✅ "Look Dev Pipeline Complete" — the full HDR/bloom/tonemap post-processing stack is done and will not need major changes again; later phases only feed new content *into* it.

**Common Mistakes:**
- Blurring at full resolution (huge, unnecessary performance cost) instead of downsampled mip levels.
- Bloom threshold set on tonemapped (LDR) data instead of HDR data, giving physically nonsensical glow.

**Testing Strategy:** Visual A/B comparisons at different bloom parameters; performance benchmarking (ms per bloom pass) at different resolutions.

**Refactoring Tasks:** Consolidate all post-fx passes behind a single `PostProcessPipeline` orchestrator class so `app.py` doesn't manually sequence passes.

**Performance Notes:** This is your first real performance-sensitive phase — profile blur pass cost at your target resolution now, since ray marching in Phase 6 will be far more expensive and you'll need this baseline for comparison.

---

### **Phase 6 — Curved Spacetime: Geodesic Ray Marching (Schwarzschild)**

**Goal:** The heart of the project. Replace the flat-space sphere/ring with a real GPU ray marcher that integrates null geodesics through Schwarzschild spacetime, producing gravitational lensing and the visual event horizon/photon sphere.

**Concepts to Learn:**
- Null geodesic equations for Schwarzschild spacetime (deriving/using the standard reduced form for photon orbits)
- Effective potential for photons, photon sphere at $r = 1.5 r_s$
- Porting an RK4 integrator from Python (Phase 1) to GLSL, including the numerical-precision differences (float32 on GPU vs float64 on CPU)
- Ray generation from camera (per-pixel view rays) in a fullscreen-quad shader
- Adaptive vs fixed step-size integration for performance/stability tradeoffs

**Files to Create:**
`physics/geodesics.py` (CPU reference implementation, used for validation), `shaders/raymarch/geodesic.frag`, `pipeline/raymarch_pass.py`, `pipeline/uniform_packing.py`

**Physics:**
- Implement the geodesic equations (in reduced/conserved-quantity form to minimize per-step cost) on the CPU first, validated against Phase 1's weak-field results and known strong-field benchmarks (e.g., photon capture cross-section $b_{crit} = 3\sqrt{3}\, r_s / 2$).
- Only after CPU validation, port to GLSL.
- Explicitly log/flag: this is a **physically accurate** geodesic integration, not an approximation — a good moment to document, in `docs/physics_notes.md`, exactly which simplifications remain (e.g., static background stars treated as being at infinity).

**Graphics:** Full-screen ray-marching fragment shader; each pixel independently integrates a photon path backward from the camera, terminating on horizon capture, disk intersection (stub for now), or escape to a background skybox.

**Estimated Difficulty:** ★★★★★ (High — the hardest single phase in the project)
**Estimated Time:** 20–35 hours (this is the phase to budget the most time and patience for)

**Expected Result:** A black hole silhouette with visible gravitational lensing distorting a background starfield/skybox, and the characteristic photon-ring bright edge.

**Milestone:** ✅ "First Real Lensed Black Hole" — this is the portfolio-defining screenshot moment.

**Common Mistakes:**
- Using Cartesian RK4 directly instead of a numerically stable formulation (e.g., using conserved energy/angular momentum to reduce the ODE system) — leads to orbit drift and visible artifacts.
- Fixed step size too coarse near the photon sphere → jagged/incorrect ring; too fine everywhere → unusable framerate. (Adaptive step size is the real answer — introduced explicitly in this phase.)
- Forgetting horizon termination condition → integrator spirals into $r=0$ and produces NaNs (defensive shader coding required).
- Not validating against the CPU reference before trusting the shader — GPU float32 precision issues near strong curvature are real and will silently produce wrong images if unchecked.

**Testing Strategy:** CPU-side pytest suite validating geodesic integration against analytic benchmarks (deflection angle, critical impact parameter, photon sphere radius) — this is your ground truth. GPU output compared visually and, where feasible, numerically (e.g., render known test rays and compare bend angle in-shader via a debug readback) against the CPU reference.

**Refactoring Tasks:** Extract step-size/adaptive-integration logic into a reusable GLSL include (`common/integrator.glsl`) since Kerr (Phase 10) will need the same scaffolding with a different metric.

**Performance Notes:** This is the performance phase. Discuss: step count vs quality tradeoff, early-ray-termination, resolution scaling (render at lower res + upscale if needed), and why this workload is embarrassingly parallel (ideal GPU fit) but still bound by worst-case per-pixel step count.

---

### **Phase 7 — Accretion Disk: Geometry, Shading & Compositing**

**Goal:** Add a physically-motivated accretion disk that the ray marcher correctly intersects (with lensing already bending the visible disk shape, including the light passing "behind" the black hole appearing above/below it — the classic Interstellar-style ring).

**Concepts to Learn:**
- Disk intersection during ray marching (plane-crossing detection mid-integration, not a simple ray-plane test, because the ray is curved)
- ISCO (Innermost Stable Circular Orbit) as the physically motivated inner disk radius
- Procedural disk texturing (noise-based turbulence, radial density falloff)
- Emissive-only shading model (the disk is a light source, not lit by an external light)

**Files to Create:**
`physics/schwarzschild.py` (extend with ISCO calculation), `scene/accretion_disk.py`, `shaders/disk/disk_shading.glsl`, integrate into `shaders/raymarch/geodesic.frag`

**Physics:** ISCO radius at $r = 6\,GM/c^2$ for Schwarzschild; disk temperature/brightness falloff approximated via a simplified radial profile (document clearly that a full radiative-transfer disk model is out of scope — this is the visual-approximation vs physically-accurate distinction called out earlier).

**Graphics:** Mid-integration plane-crossing test inside the ray marcher; procedural noise texture (via a noise LUT or in-shader noise) for disk turbulence; additive compositing of disk emission with lensed background.

**Estimated Difficulty:** ★★★★☆ (High)
**Estimated Time:** 12–18 hours

**Expected Result:** A convincing lensed accretion disk, including the "ring above and below" lensing artifact, without physically accurate radiative transfer.

**Milestone:** ✅ "Lensed Accretion Disk" — visually the project starts looking like reference images of real black hole renders.

**Common Mistakes:**
- Testing disk intersection as a simple ray-plane test against the *straight* camera ray instead of the *curved* integrated path — this silently breaks the lensed disk appearance.
- Disk edges aliasing badly without anti-aliasing/soft falloff at ISCO boundary.

**Testing Strategy:** Visual comparison against reference renders/photos (e.g., the EHT M87* image, and known Schwarzschild-lensing simulations) for qualitative sanity, plus regression screenshots to catch shader changes that break the disk.

**Refactoring Tasks:** Move disk-specific ray-marching logic into its own GLSL include so `geodesic.frag` doesn't become a monolith.

**Performance Notes:** Disk shading adds per-step cost; profile before/after and consider an early-out once a ray has already hit the disk opaquely.

---

### **Phase 8 — Gravitational Redshift & Doppler Beaming**

**Goal:** Add the two key relativistic *color/brightness* effects that make the disk look physically alive: gravitational redshift (light loses energy climbing out of the well) and Doppler beaming (the side of the disk rotating toward the camera appears brighter/blue-shifted, the receding side dimmer/red-shifted).

**Concepts to Learn:**
- Gravitational redshift factor $1+z = 1/\sqrt{1 - r_s/r}$
- Relativistic Doppler formula and beaming (the $\delta^3$ or $\delta^4$ intensity boosting factor, depending on whether you model bolometric or monochromatic intensity — worth explicitly deciding and documenting)
- Combining redshift + beaming into a single "relativistic factor" applied to disk emission color/intensity
- Blackbody-ish color mapping (temperature → RGB, simplified)

**Files to Create:**
`physics/redshift.py`, extend `shaders/disk/disk_shading.glsl`

**Physics:** Implement and unit-test the redshift/beaming formulas on CPU first (same pattern as Phase 6 — CPU validation before GPU port). This is a good phase to write a clear `docs/physics_notes.md` section distinguishing which factor you're using and why.

**Graphics:** Apply computed redshift/beaming as a color and intensity multiplier on disk fragments, feeding into the existing HDR/bloom pipeline from Phase 4–5 so bright approaching-side matter actually blooms.

**Estimated Difficulty:** ★★★☆☆ (Medium-High)
**Estimated Time:** 8–12 hours

**Expected Result:** Visibly asymmetric disk brightness/color — one side notably brighter and bluer, the other dimmer and redder — a hallmark of correct relativistic rendering.

**Milestone:** ✅ "Relativistic Disk Shading" — the disk is no longer just geometrically lensed but *physically* colored/shaded.

**Common Mistakes:**
- Applying Doppler shift based on the wrong velocity direction (need disk orbital velocity at the point of emission, projected along the photon's local direction — easy to get backwards).
- Over- or under-driving the beaming exponent, producing unrealistically extreme asymmetry (needs visual tuning against reference images, not just blind formula application).

**Testing Strategy:** Unit tests on redshift/beaming formulas at known reference radii/velocities; visual symmetry-break sanity check (front/back of disk should look correct relative to rotation direction).

**Refactoring Tasks:** Ensure redshift/beaming logic is a pure function of local physical quantities (radius, velocity, photon direction) so it's independently testable, not tangled into disk texturing code.

**Performance Notes:** Minor — a few extra scalar ops per disk-hit pixel; not a bottleneck compared to Phase 6/7.

---

### **Phase 9 — Camera Systems: Free Orbit & Cinematic Flight**

**Goal:** Elevate the camera from a functional dev tool (Phase 3) into a polished dual-mode system: free orbit/fly camera for exploration, and scripted cinematic flight paths for presentation/demo reels.

**Concepts to Learn:**
- Quaternions for smooth rotation interpolation (avoiding gimbal lock)
- Spline-based camera paths (Catmull-Rom or Bezier) for cinematic flight
- Time-based vs frame-based animation (why cinematic camera timing must be decoupled from frame rate)
- Easing functions for smooth start/stop motion

**Files to Create:**
`scene/camera.py` (extend significantly), `scene/camera_path.py`, `ui/panels/camera_panel.py`

**Physics:** None new — but this is where you might add a "physically-plausible free-fall camera" mode as a stretch goal (camera follows a timelike geodesic instead of being externally controlled) if you want an extra-credit physics tie-in.

**Graphics:** Smooth camera interpolation feeding into the existing MVP/ray-generation system from Phase 3/6 — camera changes should require zero changes to the rendering pipeline, proving the earlier decoupling was done correctly.

**Estimated Difficulty:** ★★★☆☆ (Medium)
**Estimated Time:** 8–12 hours

**Expected Result:** Smooth, cinematic fly-through sequences around the black hole, plus a polished free-orbit mode for interactive exploration, selectable via UI.

**Milestone:** ✅ "Cinematic Camera System" — good demo-reel/portfolio-video material becomes possible here.

**Common Mistakes:**
- Using Euler angles for camera orientation interpolation (gimbal lock, jittery rotation) instead of quaternions.
- Coupling cinematic path timing to frame rate instead of wall-clock time.

**Testing Strategy:** Unit tests for quaternion interpolation correctness (slerp at t=0/0.5/1 matches expected orientations); visual smoothness checks for spline paths.

**Refactoring Tasks:** If Phase 3's camera code leaked any GL-specific assumptions, this is the phase to fully purge them — camera math must remain GL-independent.

**Performance Notes:** Negligible GPU cost; ensure interpolation math doesn't allocate per-frame (watch for NumPy array churn in the hot path).

---

### **Phase 10 — Kerr Black Holes: Spin & Frame Dragging (Advanced)**

**Goal:** Generalize the Schwarzschild-only engine to support the Kerr metric — rotating black holes — introducing frame dragging and an adjustable spin parameter.

**Concepts to Learn:**
- Kerr metric in Boyer-Lindquist coordinates
- Frame dragging (Lense-Thirring effect) conceptually and mathematically
- Ergosphere vs event horizon distinction
- How the geodesic equations change with spin (additional conserved quantity: Carter's constant)
- Numerical challenges specific to Kerr integration (coordinate singularities, more complex effective potential)

**Files to Create:**
`physics/kerr.py`, extend `shaders/raymarch/geodesic.frag` (or split into `geodesic_kerr.frag` behind a metric-selection uniform/branch), extend `scene/black_hole.py` with spin parameter

**Physics:** Implement and validate Kerr geodesics on CPU first (same validated-CPU-before-GPU-port pattern as Phase 6), checking against known Schwarzschild limit (spin = 0 should reduce exactly to Phase 6's results — an excellent regression test) and known Kerr benchmarks (e.g., photon sphere radius depends on spin and orbital direction — prograde vs retrograde).

**Graphics:** Extend the ray marcher to handle the more complex Kerr geodesic equations; visually, expect an asymmetric event horizon shadow (the classic "D-shaped" Kerr silhouette) and frame-dragging-distorted disk.

**Estimated Difficulty:** ★★★★★ (Very High — the most advanced physics in the whole project)
**Estimated Time:** 25–40 hours

**Expected Result:** An adjustable spin slider that morphs the black hole shadow from circular (Schwarzschild) to the characteristic asymmetric Kerr shape in real time.

**Milestone:** ✅ "Kerr Black Hole Online" — this is genuinely advanced computational astrophysics territory; very few hobby projects reach this point.

**Common Mistakes:**
- Not validating the spin=0 limit against Schwarzschild results first — this is your most valuable regression test and skipping it makes Kerr bugs nearly impossible to diagnose.
- Numerical instability near the ergosphere/horizon without careful step-size handling.

**Testing Strategy:** Extensive CPU pytest suite: spin=0 reduces to Schwarzschild (within tolerance), known Kerr photon-sphere radii at various spins match published values, prograde/retrograde asymmetry present and correctly signed.

**Refactoring Tasks:** By now, `geodesic.frag` needs a clean metric-abstraction so Schwarzschild and Kerr paths share the integrator scaffolding (from the Phase 6 refactor) without duplicated code.

**Performance Notes:** Kerr integration is more expensive per step (more terms, more conserved quantities to track); revisit adaptive step-size tuning and consider quality presets (e.g., "cinematic" vs "interactive" step counts).

---

### **Phase 11 — Adjustable Mass/Spin UI, Presets & Scientific Modes**

**Goal:** Turn the engine into a proper interactive tool: full ImGui control panels for mass, spin, disk parameters, render quality, plus save/load presets (e.g., "Sagittarius A*", "Stellar-mass BH", "Extreme Kerr").

**Concepts to Learn:**
- imgui_bundle patterns for sliders/plots/tabs in a real-time app
- Debounced/throttled parameter updates (avoiding shader recompiles or uniform thrashing every frame)
- Preset serialization round-tripping cleanly through the Phase 2 config system

**Files to Create:**
`ui/panels/physics_panel.py`, `ui/panels/render_panel.py`, `config/presets/*.yaml` (populated with real astrophysical values)

**Physics:** Populate presets with real values (Sagittarius A* mass ≈ 4.15 million solar masses, M87* ≈ 6.5 billion solar masses, etc.) — a nice moment to connect the engine back to real astrophysics.

**Graphics:** Live-updating uniforms without stutter; UI/UX polish (tooltips explaining each physics parameter in plain language — genuinely useful given this is a teaching-oriented project).

**Estimated Difficulty:** ★★☆☆☆ (Low-Medium — mostly integration work, not new hard concepts)
**Estimated Time:** 8–12 hours

**Expected Result:** A polished, self-explanatory interactive tool a non-technical viewer could sit down and use.

**Milestone:** ✅ "Interactive Scientific Tool" — feels like a finished application, not a tech demo.

**Common Mistakes:**
- Recompiling/relinking shaders every frame when only a uniform changed (expensive; only structural changes like toggling Kerr on/off should trigger recompiles).

**Testing Strategy:** UI smoke tests where feasible (config round-trip tests are the most valuable here); manual UX pass.

**Refactoring Tasks:** Consolidate any remaining scattered "quick and dirty" debug UI from earlier phases into the proper panel system.

**Performance Notes:** Ensure ImGui rendering itself isn't a bottleneck at high panel complexity; profile UI pass separately from the raymarch pass.

---

### **Phase 12 — Polish, Packaging & Portfolio Presentation**

**Goal:** Final engineering polish pass: performance profiling sweep, documentation completion, packaging for distribution, and preparing portfolio materials (screenshots, demo video, write-up).

**Concepts to Learn:**
- Profiling methodology (CPU vs GPU bottleneck identification)
- Packaging a Python OpenGL app for distribution (dependency bundling considerations)
- Writing a compelling technical README/case-study for a portfolio audience
- Semantic versioning for a 1.0 release; CHANGELOG discipline

**Files to Create:** `docs/architecture.md` (finalized), `docs/physics_notes.md` (finalized), `CHANGELOG.md` (finalized), `README.md` (portfolio-facing rewrite)

**Physics:** Final accuracy pass — a documented table of "what's physically accurate vs visually approximated" for the whole engine, which is genuinely valuable content for a portfolio write-up and shows scientific honesty.

**Graphics:** Final quality/performance tuning across quality presets (e.g., 30fps interactive mode vs high-quality screenshot mode with higher step counts/supersampling).

**Estimated Difficulty:** ★★☆☆☆ (Low-Medium)
**Estimated Time:** 10–15 hours

**Expected Result:** A shippable, well-documented, portfolio-ready application with recorded demo footage.

**Milestone:** ✅ "v1.0 Release" — project complete.

**Common Mistakes:**
- Underselling the physics work in the README (this project's differentiator vs generic shader demos is the validated, physically-grounded geodesic integration — the write-up should say so clearly and show the validation tests).

**Testing Strategy:** Full regression pass across the entire test suite; final manual QA checklist across all UI panels/presets.

**Refactoring Tasks:** Dead code removal, dependency audit, final PEP8/type-hint sweep.

**Performance Notes:** Document final performance characteristics (resolution vs FPS table across quality presets) as part of the technical write-up.

---

## 4. Parallel Learning Roadmap

This tracks alongside the phases above — what to study *before* or *during* each phase so the implementation isn't just copy-following instructions.

| Phase(s) | Study This | Suggested Docs/Resources Type | Math to Study | OpenGL to Understand | Exercises |
|---|---|---|---|---|---|
| 0 | Python packaging, Git | Official Python packaging guide, Git branching model docs | — | — | Set up a throwaway repo and practice conventional commits |
| 1 | Special/General relativity basics, ODE numerics | A GR-for-programmers style intro; any numerical methods reference on RK4 | Vectors, derivatives, basic ODEs, RK4 derivation | — | Reproduce the weak-field light-bending formula by hand before coding it; implement RK4 for a pendulum first |
| 2 | Software architecture patterns | SOLID principles reference, dataclasses docs | — | — | Refactor a toy app into layered architecture as practice |
| 3 | OpenGL pipeline, linear algebra for graphics | ModernGL docs, "Learn OpenGL" style pipeline explanations, PyGLM docs | Matrices, perspective projection derivation, dot/cross product geometric meaning | VAO/VBO, shader stages, MVP matrix flow | Build the projection matrix by hand once before using PyGLM's helper, to understand what it does |
| 4 | HDR/color theory | Any reference on framebuffer objects, gamma correction | Basic exponential/log math for gamma | FBOs, texture formats | Deliberately break gamma correction once to *see* the visual bug, then fix it |
| 5 | Post-processing techniques | Bloom/gaussian blur explainers (conceptual, not engine-specific) | Discrete convolution basics (for blur kernels) | Multi-pass rendering, downsampling | Implement blur radius as a live-tunable ImGui slider and observe cost vs quality |
| 6 | General Relativity: geodesics, Schwarzschild metric | A solid GR textbook chapter or lecture notes on null geodesics; ray marching technique explainers | Differential equations, conserved quantities (energy/angular momentum), effective potential | Fragment-shader ray marching, per-pixel parallel computation | Plot several photon trajectories at different impact parameters with matplotlib before touching GLSL |
| 7 | Accretion disk physics (simplified) | Conceptual overviews of ISCO and accretion disks (non-radiative-transfer level is fine) | Circular orbit stability analysis (conceptual) | Mid-ray-march intersection testing | Vary ISCO radius and observe how disk inner edge should respond |
| 8 | Relativistic Doppler effect, redshift | Any SR-Doppler reference | Lorentz factor, relativistic velocity addition (conceptual) | Color/intensity mapping in shaders | Compute redshift factor by hand at a few radii and compare to your code's output |
| 9 | Quaternions, splines | Quaternion interpolation (slerp) explainers, Catmull-Rom spline references | Quaternion algebra basics | Time-based animation patterns | Animate a simple object along a spline before wiring it to the real camera |
| 10 | Kerr metric, frame dragging | GR lecture notes covering Kerr/Boyer-Lindquist coordinates | Carter's constant (conceptual), more advanced conserved-quantity ODE reduction | Metric-abstraction shader design | Re-derive the Schwarzschild limit from your Kerr equations by hand (set spin=0) before coding the regression test |
| 11 | UI/UX for scientific tools | imgui_bundle docs/examples | — | Efficient uniform update patterns | Design the panel layout on paper first |
| 12 | Technical writing, profiling | Any GPU profiling methodology reference | — | — | Write the "accuracy vs approximation" table as a study exercise, not just documentation |

---

## 5. Physics Roadmap (Summary View)

| Concept | Introduced In | Accuracy Level |
|---|---|---|
| Schwarzschild radius, escape velocity | Phase 1 | Physically accurate |
| RK4 numerical integration | Phase 1 | Physically accurate (standard technique) |
| Weak-field light deflection | Phase 1 | Physically accurate approximation (valid at large impact parameter) |
| Null geodesics (Schwarzschild) | Phase 6 | Physically accurate |
| Photon sphere, critical impact parameter | Phase 6 | Physically accurate |
| Event horizon | Phase 6 | Physically accurate (as a coordinate/causal boundary) |
| ISCO | Phase 7 | Physically accurate (as inner disk boundary) |
| Accretion disk emission profile | Phase 7 | Visual approximation (no radiative transfer) |
| Gravitational redshift | Phase 8 | Physically accurate |
| Doppler beaming | Phase 8 | Physically accurate (choice of exponent documented) |
| Kerr metric, frame dragging | Phase 10 | Physically accurate |
| Ergosphere | Phase 10 | Physically accurate |
| Camera free-fall geodesics | Phase 9 (stretch) | Physically accurate if implemented |

---

## 6. Graphics Roadmap (Summary View)

| Concept | Introduced In |
|---|---|
| GL context, clear screen | Phase 0 |
| VAO/VBO, mesh rendering, MVP matrices | Phase 3 |
| Free-fly camera | Phase 3 |
| Framebuffers, HDR, gamma | Phase 4 |
| Tonemapping | Phase 4 |
| Bloom (multi-pass) | Phase 5 |
| Fullscreen-quad shader pattern | Phase 4 (setup), Phase 6 (payoff) |
| GPU ray marching | Phase 6 |
| Mid-integration geometry intersection (disk) | Phase 7 |
| Procedural noise texturing | Phase 7 |
| Quaternion camera interpolation, spline paths | Phase 9 |
| Metric-abstracted shader branching (Kerr) | Phase 10 |
| Real-time ImGui control panels | Phase 11 |

---

## 7. Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| GPU float32 precision breaks down near strong curvature/horizon | High | High | CPU double-precision reference implementation validated first (Phase 1, 6, 10); defensive shader coding for near-singularity regions |
| Scope creep — trying to build Kerr + all effects before Schwarzschild works | Medium-High | High | Strict phase gating; Phase 6 (Schwarzschild lensing) must be fully working and validated before Phase 10 (Kerr) begins |
| Performance collapse from unbounded ray-march step counts | Medium | High | Adaptive step sizing introduced explicitly in Phase 6; quality presets in Phase 11 |
| Underestimating GR math learning curve | High | Medium | Learning roadmap paces theory *before* each implementation phase; CPU-first validation approach catches conceptual errors before they hit shader code (harder to debug) |
| Shader hot-reload / iteration friction slowing development | Medium | Medium | Build shader hot-reloading into `gl_core/shader_program.py` early (Phase 3) so later GLSL-heavy phases (6, 7, 10) iterate fast |
| Architecture erosion (physics/rendering coupling creeping in) | Medium | Medium | The Phase 2 layering is enforced by convention, not tooling — periodically audit imports; consider a simple lint rule/CI check that `physics/` never imports `gl_core/` |
| Losing motivation during the hardest phase (6 or 10) | Medium | High (project abandonment) | Phases are scoped to always end in a visible milestone; Phase 6 specifically front-loads CPU/matplotlib validation so you *see* correct-looking geodesics well before the GPU port, providing confidence checkpoints along the way |
| ModernGL/moderngl-window/imgui_bundle version incompatibilities | Low-Medium | Medium | Pin dependency versions in `pyproject.toml` from Phase 0; document tested versions in README |

---

## 8. Recommended Development Order

The phase numbers above **are** the recommended order — they're designed as a dependency chain, not a menu. A few explicit notes on why:

1. **0 → 1 → 2** must happen in order but are relatively fast; they're foundation-laying, not feature work.
2. **3 → 4 → 5** build the entire rendering pipeline in flat space *before* anything relativistic touches it. This is deliberate: debugging HDR/bloom bugs is much easier without curved-spacetime ray marching also in the mix.
3. **Phase 6 is the pivot point of the entire project.** Do not rush it, and do not skip the CPU-validation step before writing GLSL — this is the single highest-leverage risk-mitigation step in the whole roadmap.
4. **7 → 8** naturally follow 6 since they both extend the ray marcher incrementally.
5. **Phase 9** (camera polish) is intentionally placed *after* the hard physics phase (6) rather than earlier, so you're polishing camera feel against the real lensed renderer, not a placeholder scene.
6. **Phase 10 (Kerr)** is explicitly gated behind a fully validated Phase 6 — attempting Kerr before Schwarzschild is solid is the most common way this kind of project stalls out.
7. **11 → 12** are integration/polish and can partially overlap with late Phase 10 work if you want a Schwarzschild-only "release" milestone before tackling Kerr, which is a very reasonable scope-management option if time is limited — Phase 9's end state is already a complete, portfolio-worthy Schwarzschild simulator on its own.

**Optional off-ramp:** If at any point Kerr (Phase 10) feels like too much, stopping after Phase 9 still yields a complete, scientifically grounded, visually polished Schwarzschild black hole simulator — a fully valid place to call the core project "done" and treat Phases 10+ as a stretch/v2 goal.

---

## Next Step

This document is the map, not the territory. When you're ready to start building, say **"Next Phase"** and we'll begin with **Phase 0**: theory first (why each tooling choice matters), then the actual files, then a verification step, before moving on.
