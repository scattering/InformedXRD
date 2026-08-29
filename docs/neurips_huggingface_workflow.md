# NeurIPS Hugging Face Workflow

This note documents the planned artifact workflow for the NeurIPS Evaluations & Datasets submission. It is intentionally separate from the PRX archival release.

## Goals

- Keep the NeurIPS release surface distinct from the PRX public repo and Zenodo archive.
- Use GitHub for code, manifests, docs, and compact paper-backed outputs.
- Use Hugging Face for larger binary artifacts needed for NeurIPS review/release.
- Preserve double-blind hygiene during review.

## Repositories and Roles

### Private working repo

- Repo: `project`
- Purpose:
  - planning
  - staging
  - internal scripts
  - release checklists
  - artifact assembly

### Public GitHub repo

- Repo: `scattering/neurips-project`
- Purpose:
  - evaluation code
  - benchmark construction code
  - manifests
  - compact JSON/CSV artifacts
  - reviewer-facing docs

### Hugging Face dataset repo

- Purpose:
  - `RRUFF-473` / `RRUFF-325` payloads if released directly
  - reviewer subset
  - benchmark metadata
  - dataset card

### Hugging Face model repo

- Purpose:
  - NeurIPS case-study checkpoints only
  - model metadata
  - model card

### PRX release surface

- PRX public repo / PRX archival package remain the broader companion-paper release.
- Do not treat the NeurIPS HF artifacts as a mirror of the PRX Zenodo archive.

## Local Staging Directories

Current local staging roots:

- `/Users/UserE/neurips-hf-dataset-staging`
- `/Users/UserE/neurips-hf-model-staging`

These are local-only staging areas and should be treated as scratch release assembly surfaces until the actual HF repos are created.

## Planned Artifact Split

### Keep in GitHub

- benchmark construction code
- extinction-group mapping utilities
- hierarchy-aware metric code
- prior-only baseline code
- stratified evaluation code
- compact paper-backed outputs
- reviewer docs
- manifests:
  - checkpoint manifest
  - dataset manifest
  - artifact manifest

### Put in Hugging Face dataset repo

- benchmark payloads for `RRUFF-473` and `RRUFF-325`, if redistributed directly
- reviewer-ready subset
- benchmark metadata and checksums
- dataset card text

### Put in Hugging Face model repo

- four ViT case-study checkpoints used in the NeurIPS paper
- one 1D residual CNN control checkpoint
- minimal checkpoint-role metadata
- model card text

### Keep out of the NeurIPS HF release

- unused PRX checkpoints
- broad PRX archival assets
- large training corpora not needed for the NeurIPS paper
- private paths, cluster state, exploratory outputs

## Review-Time Anonymization Plan

### Default posture

- Stage artifacts privately first.
- Do not expose personal or group-identified Hugging Face repos to reviewers.
- Prefer an anonymized OpenReview supplementary bundle for review if there is any doubt.

### If Hugging Face is used during review

Use neutral names only:

- no personal usernames
- no lab/org names
- no `scattering`
- no PRX-facing naming

Review artifact repos must avoid:

- author names
- affiliations
- identifying URLs
- PRX repo links
- PRX Zenodo links
- workstation paths
- W&B links
- commit history with identifying metadata

### Things to scrub before any review-time upload

- README text
- dataset/model cards
- filenames
- notebook metadata
- checkpoint metadata/config payloads
- CSV/JSON provenance fields
- local paths in configs

## Publication Sequence

1. Assemble artifacts in the private repo and local HF staging directories.
2. Verify manifests against actual files.
3. Create HF dataset/model repos as private.
4. Upload artifacts privately and test download/load paths.
5. For review:
   - use anonymized supplementary package, or
   - use neutral anonymized HF repos if strictly necessary.
6. After acceptance:
   - switch to final public naming
   - publish HF repos
   - link them from the public GitHub repo

## Open Questions

- whether full benchmark payloads will be redistributed, or only reviewer subsets plus reconstruction manifests
- whether the review artifact will rely on OpenReview supplementary only, or also on anonymized HF hosting
- whether UserC's extinction-group figure notebook should be converted into a clean script before inclusion in the public repo
