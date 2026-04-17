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
* Updated the cookiecutter template to use the latest commit. (#102):
    * Added support and CI builds for Python 3.13 and 3.14.
    * Migrated `optional-dependencies` to `dependency-groups`.
    * Set token-based workflows to run within an `automation` environment.
    * Updated dependencies.
    * `pre-commit` has been replaced by `prek`.
    * `Makefile` now handles some dependency installation logic.
    * `tox` now uses the new TOML spec for configurations.

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
