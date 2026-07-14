"""Combined runner for the phi-consuming analysis stages — load phi ONCE.

Each phi stage (evaluate_detectors, representation_ablation, severity,
reliability, cross_corruption, evaluate_mdd, analyse) is normally launched as
its own process, and each independently re-reads every run's phi from disk. At
full scale that is the same ~90 GB of phi read 5-7 times over.

This script runs those stages **inside one process sharing a single resident
loader**, so each run's phi is read from disk exactly once and reused by every
stage. It does NOT modify any stage script: it builds one shared loader and
monkeypatches the loader factory in each stage module's namespace, then calls
each stage's `main()` in dependency order.

Two loader interfaces exist in the codebase and BOTH are served from the same
in-RAM arrays here:
  * LazyFeatures  (load_all_features)  -> fit_detectors / evaluate_detectors /
                                          representation_ablation / severity /
                                          reliability / cross_corruption
  * LazyPhiDict                        -> evaluate_mdd / analyse
The LazyPhiDict consumers get a thin VIEW over the shared LazyFeatures, so phi
is never duplicated between the two interfaces.

Usage (from repo root, venv active):
    python analysis/run_phi_stages.py                  # all stages, full residency
    python analysis/run_phi_stages.py --fast           # quick smoke test
    python analysis/run_phi_stages.py --stages eval_detectors mdd analyse  # only these
    python analysis/run_phi_stages.py --skip analyse reliability           # all but these
    python analysis/run_phi_stages.py --max-resident 8 # cap RAM (LRU; clean pinned)

Prerequisite: the producer stages (extract, offline features, fusion, ANN
baselines) must already have written their artifacts — this runner only covers
the phi-scoring/fit stages, exactly the ones that re-read phi.
"""
import sys
import argparse
import os
import re
import importlib
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))


def _preparse_output_dir():
    """Resolve --output-dir BEFORE importing benchmark_config: that module reads
    VMEM_OUTPUT_DIR at import and derives every path (PHI_DIR, DETECTOR_DIR,
    TABLE_DIR, results/, ...) from it, so the override must be in the environment
    first or those constants (some computed at import time) would be stale."""
    import argparse as _ap
    p = _ap.ArgumentParser(add_help=False)
    p.add_argument("--output-dir", type=str)
    a, _ = p.parse_known_args()
    if a.output_dir:
        os.environ["VMEM_OUTPUT_DIR"] = str(Path(a.output_dir).resolve())


_preparse_output_dir()

from vmem_benchmark import benchmark_config as cfg
from analysis import representation_ablation as ra
from analysis import vmem_utils
from analysis.vmem_utils import load_phi_seq_lens, load_phi_spatial, load_pt, FAST_MODE

# Default fraction of *available* host RAM the resident φ cache may occupy. The
# rest is headroom for each stage's own transient allocations (fits, GPU staging
# copies, sklearn temporaries), so a full cache never wedges the node. Tunable
# via --ram-frac.
RAM_SAFETY_FRAC = 0.85


def _avail_ram_bytes():
    """Best-effort available host RAM, no extra dependency. None if undetectable."""
    import os
    try:
        names = getattr(os, "sysconf_names", {})
        if "SC_AVPHYS_PAGES" in names and "SC_PAGE_SIZE" in names:   # Linux / macOS
            return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        pass
    try:
        import ctypes                                                # Windows
        if sys.platform == "win32":
            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = _MS(); m.dwLength = ctypes.sizeof(_MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return int(m.ullAvailPhys)
    except Exception:
        pass
    return None


def _phi_footprint():
    """Resident bytes each run's φ will occupy (materialized float32), read from
    the .pt headers via mmap so no tensor data is faulted in. Returns
    (total_bytes, mean_bytes_per_run, n_runs)."""
    sizes = []
    for f in sorted(cfg.PHI_DIR.glob("*.pt")):
        try:
            d = load_pt(f)                       # mmap: only the pickle header is read
            sizes.append(d["phi"].numel() * 4)   # numel uses shape metadata, not data
        except Exception:
            pass
    if not sizes:
        return 0, 0, 0
    return sum(sizes), sum(sizes) // len(sizes), len(sizes)


def _gb(b):
    return b / 1e9


# ─────────────────────────────────────────────────────────────────────────────
# Pretty console output. The stage scripts emit a lot of repeated lines (e.g.
# "[CPU] ... Mahalanobis fit" once per representation per run) and uncoloured
# text. We wrap *this runner's* stdout (not the stage scripts, not the shared
# device_for) with a filter that collapses consecutive duplicate lines and adds
# a light colour scheme. tqdm writes to stderr, so its live bars are untouched.
# ─────────────────────────────────────────────────────────────────────────────
class _C:
    RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
    RED = "\033[31m"; GREEN = "\033[32m"; YELLOW = "\033[33m"
    BLUE = "\033[34m"; CYAN = "\033[36m"


def _enable_ansi():
    """Enable ANSI colour on Windows terminals (VT processing). No-op elsewhere."""
    if sys.platform != "win32":
        return True
    try:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.GetStdHandle(-11)                       # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        k.GetConsoleMode(h, ctypes.byref(mode))
        k.SetConsoleMode(h, mode.value | 0x0004)      # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        return True
    except Exception:
        return False


class _PrettyWriter:
    """stdout wrapper: collapse consecutive duplicate lines + colourise."""
    _BRACKET = re.compile(r"(\[[^\]]*\])")

    def __init__(self, stream, color=True):
        self._s = stream
        self._color = color
        self._buf = ""
        self._last = None     # last raw (stripped) line, for dedup
        self._reps = 0        # how many extra times _last has repeated

    def write(self, text):
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._emit(line)
        return len(text)

    def _emit(self, line):
        key = line.strip()
        if key and key == self._last:
            self._reps += 1
            return
        self._flush_reps()
        self._last = key or None
        self._s.write(self._paint(line) + "\n")

    def _flush_reps(self):
        if self._reps > 0:
            msg = f"      ... (last line repeated x{self._reps + 1})"
            self._s.write((_C.DIM + msg + _C.RESET if self._color else msg) + "\n")
            self._reps = 0

    def _paint(self, line):
        if not self._color or not line.strip():
            return line
        low = line.lower()
        if re.search(r"fail|error|traceback|exception", low):
            return _C.RED + line + _C.RESET
        if re.search(r"warning|\[!\]|\[skip\]", low):
            return _C.YELLOW + line + _C.RESET
        if re.search(r"saved|wrote|complete|finished|\bok\b", low):
            return _C.GREEN + line + _C.RESET
        s = line.lstrip()
        if s.startswith((">>", "#", "=")):            # runner banners / stage headers
            return _C.BOLD + _C.CYAN + line + _C.RESET
        return self._BRACKET.sub(_C.CYAN + r"\1" + _C.RESET, line)   # highlight [tags]

    def flush(self):
        # NOTE: do NOT emit the repeat-summary here. Callers (device_for) print
        # with flush=True after every line, so flushing reps here would defang
        # the dedup. Reps are emitted when a *distinct* line arrives, or at
        # teardown via flush_pending(). Keep partial (newline-less) text buffered.
        self._s.flush()

    def flush_pending(self):
        """Emit any trailing repeat-summary and partial line. Call at teardown."""
        self._flush_reps()
        if self._buf:
            self._s.write(self._buf)
            self._buf = ""
        self._s.flush()

    def __getattr__(self, name):     # delegate isatty/fileno/... to the real stream
        return getattr(self._s, name)


class PhiDictView:
    """LazyPhiDict-compatible view over a shared LazyFeatures instance.

    Serves phi from the SAME arrays the LazyFeatures consumers use (no second
    copy). Implements the full surface the LazyPhiDict consumers touch:
    Mapping protocol + get_seq_lens + get_phi_spatial. phi_spatial (only the MDD
    reads it) is loaded lazily once and cached here."""

    def __init__(self, feats):
        self._f = feats
        self._spatial = {}

    def __getitem__(self, run):
        return self._f[run]["phi"]

    def __contains__(self, run):
        return run in self._f

    def __iter__(self):
        return iter(self._f)

    def __len__(self):
        return len(self._f)

    def keys(self):
        return self._f.keys()

    def get(self, run, default=None):
        return self[run] if run in self else default

    def get_seq_lens(self, run):
        return load_phi_seq_lens(run)

    def get_phi_spatial(self, run):
        if run not in self._spatial:
            self._spatial[run] = load_phi_spatial(run)
        return self._spatial[run]


# Stage registry: name -> (module path, dependency-order rank). Producer stages
# (extract/offline/fusion/ann) are intentionally absent — they create artifacts
# rather than re-reading phi. fit_detectors runs first so evaluate_detectors and
# severity find the freshly written detectors + split.json.
STAGES = [
    ("fit_detectors",         "analysis.fit_detectors"),
    ("eval_detectors",        "analysis.evaluate_detectors"),
    ("mdd",                   "analysis.evaluate_mdd"),
    ("representation_ablation", "analysis.representation_ablation"),
    ("severity",              "analysis.severity"),
    ("reliability",           "analysis.reliability"),
    ("cross_corruption",      "analysis.cross_corruption"),
    ("analyse",               "analysis.analyse"),
]

# Modules that did `from analysis.representation_ablation import load_all_features`
# — each holds its OWN binding of that name, so the patch must be applied in each
# module's namespace (rebinding only ra.load_all_features would miss them).
_LAF_CONSUMERS = [
    "analysis.fit_detectors", "analysis.evaluate_detectors",
    "analysis.severity", "analysis.reliability", "analysis.cross_corruption",
]
# Modules that did `from analysis.vmem_utils import LazyPhiDict`.
_LPD_CONSUMERS = ["analysis.evaluate_mdd", "analysis.analyse"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", choices=[s[0] for s in STAGES],
                    help="Only run these stages (default: all). Whitelist.")
    ap.add_argument("--skip", nargs="+", choices=[s[0] for s in STAGES], default=[],
                    help="Run every stage EXCEPT these. Blacklist; applied after "
                         "--stages, so you can combine them.")
    ap.add_argument("--max-resident", type=int, default=None,
                    help="Cap how many runs stay resident (LRU; 'clean' is always "
                         "pinned). Overrides the RAM-aware default. Lower it if "
                         "RAM-constrained — stages then re-read evicted runs.")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Run directory holding phi/ and receiving all outputs "
                         "(detectors/, tables/, results/, plots/). Rebases every "
                         "path; resolved before config import.")
    ap.add_argument("--ram-frac", type=float, default=RAM_SAFETY_FRAC,
                    help="Fraction of AVAILABLE host RAM the resident phi cache may "
                         "use (rest is headroom for stage temporaries). Higher = "
                         "more runs held in RAM = fewer disk re-reads.")
    ap.add_argument("--vram-frac", type=float, default=0.4,
                    help="Fraction of FREE VRAM each GPU chunk is sized to. Higher = "
                         "bigger chunks = better GPU saturation; chunked_apply "
                         "auto-halves on OOM, so it self-tunes to the largest fit.")
    ap.add_argument("--no-color", action="store_true",
                    help="Disable ANSI colour / line-dedup pretty output.")
    # parse_known_args so flags the stages read directly (e.g. --fast) stay in argv.
    args, _ = ap.parse_known_args()

    # Pretty console: colourise + collapse repeated lines. Auto-off when stdout
    # is redirected (keeps log files clean) or --no-color is given.
    use_color = (not args.no_color) and sys.stdout.isatty() and _enable_ansi()
    _real_stdout = sys.stdout
    # Never let a stray non-ASCII char from a stage crash the run on a legacy
    # (cp1252) Windows console — degrade that character instead of raising.
    try:
        _real_stdout.reconfigure(errors="backslashreplace")
    except Exception:
        pass
    sys.stdout = _PrettyWriter(_real_stdout, color=use_color)

    # Push the VRAM dial into the shared chunking helper (every stage's GPU loops
    # route through it). Self-tuning + OOM-safe, so this only sets the start size.
    vmem_utils.set_vram_chunk_frac(args.vram_frac)

    n_runs = len(list(cfg.PHI_DIR.glob("*.pt")))
    if n_runs == 0:
        print(f"No phi files in {cfg.PHI_DIR}. Run extract.py first.")
        return

    # ── RAM-aware residency: full residency holds every run's φ in HOST RAM, so
    #    a naive default can OOM a constrained node. Estimate the footprint and,
    #    unless --max-resident is given, auto-cap residency to a safe fraction of
    #    available RAM. (This bounds the φ cache; each stage still allocates its
    #    own transient RAM, which the safety fraction reserves headroom for.) ──
    total_b, mean_b, n_sized = _phi_footprint()
    avail_b = _avail_ram_bytes()
    note = ""
    if args.max_resident is not None:
        cache_size = args.max_resident
        note = "explicit --max-resident"
    elif avail_b and mean_b:
        budget = avail_b * args.ram_frac
        fit_runs = max(1, int(budget // mean_b))     # how many runs fit the budget
        if fit_runs >= n_runs:
            cache_size, note = n_runs, "full residency fits RAM budget"
        else:
            cache_size = fit_runs
            note = (f"AUTO-CAPPED to fit {args.ram_frac:.0%} of "
                    f"{_gb(avail_b):.0f} GB available")
    else:
        cache_size = n_runs
        note = "could not detect RAM — holding all; set --max-resident if it OOMs"

    print("=" * 64)
    print("  COMBINED PHI PIPELINE — single shared resident loader")
    print("=" * 64)
    print(f"  output dir        : {cfg.OUTPUT_DIR}")
    print(f"  vram chunk frac   : {args.vram_frac:.2f} of free VRAM/chunk "
          f"(self-tuning, OOM-safe)")
    print(f"  phi files found   : {n_runs}")
    if total_b:
        print(f"  phi footprint     : ~{_gb(total_b):.1f} GB total "
              f"(~{_gb(mean_b):.2f} GB/run, float32 resident)")
    print(f"  host RAM available: "
          f"{(f'~{_gb(avail_b):.0f} GB' if avail_b else 'unknown')}")
    print(f"  resident cache    : {cache_size} run(s) "
          f"({'full - each run read once' if cache_size >= n_runs else 'LRU, clean pinned'})"
          f"  [{note}]")
    if cache_size < n_runs:
        print(f"  [!] residency < runs: evicted runs are re-read from disk (mmap). "
              f"Raise --max-resident if you have more RAM.")
    if FAST_MODE:
        print("  --fast            : ON (smoke-test subsample)")

    # ── one shared loader, injected into every stage ──────────────────────────
    shared = ra.LazyFeatures(cache_size=cache_size, verbose=True)
    view = PhiDictView(shared)

    def laf_factory(*a, **k):
        return shared

    def lpd_factory(*a, **k):
        return view

    # Patch every consumer's own binding (modules are imported on demand below,
    # so import them first, then rebind the name in each namespace).
    for modname in _LAF_CONSUMERS + _LPD_CONSUMERS + ["analysis.representation_ablation"]:
        importlib.import_module(modname)
    ra.load_all_features = laf_factory
    for modname in _LAF_CONSUMERS:
        sys.modules[modname].load_all_features = laf_factory
    for modname in _LPD_CONSUMERS:
        sys.modules[modname].LazyPhiDict = lpd_factory

    order = {name: i for i, (name, _) in enumerate(STAGES)}
    selected = set(args.stages or [s[0] for s in STAGES])   # whitelist (default all)
    selected -= set(args.skip)                              # then drop the blacklist
    selected = sorted(selected, key=lambda n: order[n])     # keep dependency order
    stage_mod = dict(STAGES)

    if not selected:
        print("  No stages left to run after --stages/--skip filtering.")
        return
    if args.skip:
        print(f"  skipping          : {', '.join(args.skip)}")
    print(f"  stages            : {', '.join(selected)}")
    print("=" * 64)

    results = []
    try:
        for i, name in enumerate(selected, 1):
            mod = sys.modules[stage_mod[name]]
            print(f"\n>> [{i}/{len(selected)}] {name}", flush=True)
            t0 = time.time()
            try:
                mod.main()
                status = "ok"
            except SystemExit as e:      # stages call sys.exit() on missing inputs
                status = f"exit({e.code})"
            except Exception as e:       # one stage failing must not abort the rest
                import traceback
                traceback.print_exc()
                status = f"FAILED: {type(e).__name__}"
            dt = time.time() - t0
            results.append((name, status, dt))
            tag = "ok" if status == "ok" else status
            print(f"   {name} -> {tag} in {dt:.1f}s", flush=True)

        print("\n" + "=" * 64)
        print("  SUMMARY")
        print("=" * 64)
        for name, status, dt in results:
            print(f"  {name:24s} {status:20s} {dt:8.1f}s")
        total = sum(dt for _, _, dt in results)
        print(f"  {'TOTAL':24s} {'':20s} {total:8.1f}s")
        print(f"  phi loaded once into a shared loader ({cache_size}-run residency).")
    finally:
        sys.stdout.flush_pending()
        sys.stdout = _real_stdout


if __name__ == "__main__":
    main()
