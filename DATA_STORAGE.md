# Reference downloads and project-generated data

Downloaded MaleCNS reference files stay in:

`/Users/monomyth/code/data/malecns/v1.0`

All Codex-generated datasets, model checkpoints, processed circuits, feature
caches, exports, brain layouts and run records belong in:

`/Users/monomyth/code/codex/fly-brain/data`

The project data directory's `v1.0` link points to the downloaded references;
it does not duplicate them. Files belonging to other projects were not moved.
The relocation record is in `reports/pickup-fix/storage-migration.json`.

The CLI and simulator UI now default to the project directory. Set
`FLY_BRAIN_DATA_HOME=/Users/monomyth/code/codex/fly-brain/data` explicitly if
needed. Legacy `--home` values pointing at the old shared root are redirected
for compatibility; temporary portable bundles can still supply their own root.
Legacy generated-data paths passed through the CLI are also redirected.

Immutable checkpoint and dataset manifests retain their original paths as
historical provenance, so their recorded file hashes are preserved. Current
`reports/*-path.txt` pointers identify the relocated files.

Other projects can reference or export a checkpoint from this project directory.
Generated files are not placed back in the shared download directory.
