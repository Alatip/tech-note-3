Data files are gitignored. `./fetch.sh` downloads the two Chen 2026 matrices used
(MMLU-Pro + MATH-500 in `matrix_marketE2_final.json`, MATH-Hard in
`matrix_marketMH_final.json`, ~7 MB) from HF `josefchen/co-failure-67-models`
(CC-BY 4.0, snapshot sha f2dd304420cfe48bf8e5c10dcb6735ec3dd5252c, 2026-09-25).

D4 (Open LLM Leaderboard v1, 28 base models x 14,042 MMLU items) is read from a
per-item cache in tech-note-1 format; see the top-level README.
