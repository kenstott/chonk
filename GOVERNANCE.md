# Governance

Chonk is maintained by a single maintainer, Kenneth Stott
([@kenstott](https://github.com/kenstott)), who has final say on design
decisions, merges, and releases.

## Decision making

- Proposals for significant changes (new public API, new extension point,
  storage schema change, new core dependency) start as a GitHub issue so the
  design can be discussed before code is written.
- The maintainer reviews every pull request. A pull request is merged when it
  passes CI and the maintainer approves it.

## Releases

- Versions follow Semantic Versioning. Before 1.0, a minor bump may break the API;
  breaking changes are listed in `CHANGELOG.md`.
- A release is cut by pushing a `v*` tag matching the version in
  `pyproject.toml`. The release workflow runs the test suite, including the
  PostgreSQL backend lane, and publishes to PyPI.

## Becoming a maintainer

Contributors with a sustained record of accepted pull requests may be invited to
become maintainers with merge rights. Once there is more than one maintainer,
this document will be updated to describe how they reach decisions.

## Continuity

If the maintainer can no longer maintain the project, the repository will be
archived or transferred to a willing contributor, and the decision will be
announced in the README.
