# iOS release source tracking

For every build, commit and push the relevant reviewed source to origin on the established branch. Preserve unrelated working changes. Never force-push release history.

Before archiving, record the source commit and a SHA-256 source manifest. Keep build-number and signing overrides explicit. Run the relevant tests, review the staged diff, and exclude credentials, private logs, tester records, DerivedData, archives and IPA files. Verify the pushed remote SHA before describing the source as backed up.

Build 10 was archived before this initial iOS source commit. Its release record and manifest document retrospective equivalence: 203 files match exactly; the Xcode project differs only in Release build number, signing settings and dSYM generation. The manifest stores the archived project hash, not the default development project hash. One incidental Python bytecode entry was excluded. This commit must not be described as the original archive commit.

The frozen report resources and selected public contract fixtures are intentional. Preserve their bytes; do not renew their timestamps. Some older tests contain machine-specific optional fixture paths; their skips do not establish cross-machine coverage. Build 10 ran 592 tests with two skips and zero failures. Manual four-sport Analytics visual acceptance remained incomplete when release was authorized. Terms changes were excluded.
