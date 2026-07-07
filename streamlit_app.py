"""Root-level Streamlit entry point — the actual `streamlit run` target.

Closes a real, live-reproduced bug class structurally rather than by
convention: a script that is the literal `streamlit run` target
re-executes its entire top-level code (including every `class` statement)
from scratch on every single rerun, so an instance of a class *defined
inside that script* (e.g. `src/main.py`'s own `PipelineStage` enum, if it
were used directly) stored in `st.session_state` becomes a different,
non-equal object once the next rerun redefines the class — a real
`KeyError`/equality-check crash, reproduced directly via
`streamlit.testing.v1.AppTest` (see Architecture_North_Star.md §3's
"Known limitation" and CLAUDE.md's repo-structure note for the full
finding). `src/main.py`'s own workaround (storing `PipelineStage.X.value`
strings, never the enum member itself, in `st.session_state`) remains in
place — this wrapper does not remove that workaround, since `main.py` can
still be run directly during development — but running the app through
this file instead means `main` is a normally `import`ed module, cached
once in `sys.modules` rather than re-executed, so this whole class of bug
cannot occur for anything imported from `agents/`, `data/`, etc. either.

Usage: `streamlit run streamlit_app.py` (not `streamlit run src/main.py`).
"""

from main import main

if __name__ == "__main__":
    main()
