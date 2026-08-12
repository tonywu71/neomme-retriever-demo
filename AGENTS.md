# Repository instructions

- Use atomic conventional commits. Every commit pushed to `main` must leave the Space deployable because the sync workflow runs on each push.
- Commit and push directly to `main`; this is a single-maintainer demo repo, no feature branches or PRs required.
- Validate the YAML block at the top of `README.md` before committing:
  - `short_description` must contain at most 60 characters.
  - `emoji` must be one emoji.
  - `colorFrom` and `colorTo` must be `red`, `yellow`, `green`, `blue`, `indigo`, `purple`, `pink`, or `gray`.
  - `sdk` must be `gradio`, `docker`, or `static`. Keep `sdk_version`, `python_version`, and `app_file` valid for that SDK.
- Keep GitHub as the source of truth. Changes made in the Space UI are overwritten by the next sync.
- Put Space dependencies in `requirements.txt` and local-only dependencies in `requirements-local.txt`.
- Never commit tokens or API keys. Store them in GitHub or Space secrets.
- Before pushing, run `python3 -m py_compile app.py theme.py vlm.py smoke_test.py`, `.venv-space/bin/python smoke_test.py`, and `git diff --check`.
- Changes that only touch `.github/**` do not trigger a sync. Run the workflow manually when testing workflow-only changes.
