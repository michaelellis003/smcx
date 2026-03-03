# Publishing Your Package

This guide covers publishing your Python package to PyPI, TestPyPI,
and conda-forge.

## PyPI (Trusted Publishing)

The release workflow uses [trusted publishing (OIDC)](https://docs.pypi.org/trusted-publishers/)
— no API tokens or passwords are needed. GitHub Actions authenticates
directly with PyPI using OpenID Connect.

### Setup

1. **Create a PyPI account** at [pypi.org/account/register](https://pypi.org/account/register/).
2. **Add a trusted publisher** at
   [pypi.org/manage/account/publishing](https://pypi.org/manage/account/publishing/):
    - **Owner**: your GitHub username or organization
    - **Repository**: your repository name
    - **Workflow name**: `release.yml`
    - **Environment**: leave blank
3. **Enable publishing** — PyPI publishing is already configured in
   `.github/workflows/release.yml`.

Once configured, every merge to `main` that includes a `feat:` or
`fix:` commit will automatically version-bump, build, and publish
to PyPI.

### Build Attestations

When PyPI publishing is enabled, the workflow also generates
[build attestations](https://docs.pypi.org/attestations/) and
[SLSA build provenance](https://slsa.dev/) via
`actions/attest-build-provenance`. This lets users verify that
your published package was built from your repository.

## Release Token

The release workflow works **out of the box** using the default
`GITHUB_TOKEN` — no extra secrets are required.

However, commits created by `github-actions[bot]` (via `GITHUB_TOKEN`)
[do not trigger other workflows](https://docs.github.com/en/actions/security-for-github-actions/security-guides/automatic-token-authentication#using-the-github_token-in-a-workflow).
This means the docs-deploy workflow won't run for the version-bump
commit.

To enable downstream workflows, create a **fine-grained Personal Access
Token** (PAT) with **Contents: read/write** scope and add it as a
repository secret named `RELEASE_TOKEN`. The release workflow will
prefer `RELEASE_TOKEN` when available and fall back to `GITHUB_TOKEN`
otherwise.

## Build Validation

Every CI run includes a `build` job that:

1. Builds the sdist and wheel with `uv build`
2. Validates package metadata with `twine check`
3. Installs the wheel in an isolated venv and verifies the import works

This catches packaging issues (missing files, bad metadata, broken
imports) before they reach a release.

## TestPyPI

Use TestPyPI to verify your publishing pipeline without affecting
the real package index.

### Setup

1. **Create a TestPyPI account** at
   [test.pypi.org/account/register](https://test.pypi.org/account/register/).
2. **Add a pending publisher** at
   [test.pypi.org/manage/account/publishing](https://test.pypi.org/manage/account/publishing/):
    - **Owner**: your GitHub username or organization
    - **Repository**: your repository name
    - **Workflow name**: `test-publish.yml`
    - **Environment**: leave blank
3. **Uncomment** the publish step in `.github/workflows/test-publish.yml`.
4. **Trigger manually** from the Actions tab → "Test Publish" → "Run workflow".

### Installing from TestPyPI

```bash
pip install --index-url https://test.pypi.org/simple/ smcjax
```

## conda-forge

A recipe skeleton is provided in `recipe/meta.yaml`. To submit your
package to [conda-forge](https://conda-forge.org/):

### Prerequisites

- Your package must already be published on PyPI.
- The `recipe/meta.yaml` file contains your package name, description,
  and maintainer (set during initialization).

### Steps

1. **Update the recipe version and SHA256** in `recipe/meta.yaml`:
    - Set `version` to the version you published on PyPI.
    - Download the sdist tarball from PyPI and compute its SHA256:
      ```bash
      curl -sL https://pypi.org/packages/source/s/smcjax/smcjax-1.0.1.tar.gz \
          | shasum -a 256
      ```
    - Replace `REPLACE_WITH_SHA256` with the computed hash.

2. **Add runtime dependencies** to the `run` section if your package
   has any (they must use conda package names, which sometimes differ
   from PyPI names).

3. **Fork [conda-forge/staged-recipes](https://github.com/conda-forge/staged-recipes)**
   on GitHub.

4. **Copy** your `recipe/meta.yaml` into `recipes/smcjax/meta.yaml`
   in the fork.

5. **Open a pull request** against `conda-forge/staged-recipes`. The
   conda-forge CI will build and validate your recipe.

6. Once merged, conda-forge creates a dedicated feedstock repository
   and your package becomes available via:
   ```bash
   conda install -c conda-forge smcjax
   ```
