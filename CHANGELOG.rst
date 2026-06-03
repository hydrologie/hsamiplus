=========
Changelog
=========


`Unreleased <https://github.com/hydrologie/hsamiplus>`_ (latest)
----------------------------------------------------------------

Contributors: Trevor James Smith (:user:`Zeitsperre`).

Changes
^^^^^^^
* Updated the cookiecutter template to use the latest commit. (#23):
    * Updated GitHub Actions and Python dependencies.
    * Added `pre-commit` hooks for `vulture` (find dead code), `codespell` (find typos), `zizmor` (GitHub workflow security), and `gitleaks` (token commit prevention).
    * Disabled coveralls for Python3.13.
* Updated the cookiecutter template to use the latest commit. (#106):
    * Replaced `pre-commit` tool with `prek`.
    * Development and documentation dependencies are now installed via `dependency-groups` (requires `pip >=25.2`).
    * `tox` now uses the TOML spec for configuration.
    * `Makefile` now handles more dependency management operations.
    * Added generative AI contribution information and model guidance files (`AGENTS.md` and `AI_POLICY.md`).
* `hsamiplus` now has guidance documents on acceptable usages of AI and the expected methods of AI usage disclosure. See the documentation for more details. (#114).
* Updated the cookiecutter template to use the latest commit. (#114):
    * Adjusted the permissions for some workflows to address security issues.
    * Added the new "standard" AI disclosure guidance for code contributions.
    * Updated the ReadTheDocs configuration to use newer OS and conda images.
    * Modified ``make servedocs`` to use `sphinx-autobuild` (``make livehtml``).
    * Added guidance for maintainers on git commit signing and immutable releases.
    * Adjusted the source distribution inclusion/exclusion list.
    * Set `bump-my-version` to sign tags by default.

Fixes
^^^^^
* No change.

.. _changes_0.1.0:

`v0.1.0 <https://github.com/hydrologie/hsamiplus/tree/0.1.0>`_
--------------------------------------------------------------

Contributors: Didier Haguma (:user:`dhagum`)

Changes
^^^^^^^
* First release on PyPI.
