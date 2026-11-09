# Manuscript sources

`part1.md` to `part4.md` and `supplement.md` are the sources; `python3 build.py` joins the parts, numbers the citations from `refs.py` and writes `manuscript.md`; then
`pandoc manuscript.md --reference-doc=ref.docx -o EviAlign-CXR_manuscript.docx` (and the same for `supplement.md`).
`[R]` marks a result that is still to be filled from `results/`.
