# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

## Types of Contributions

### Report Bugs

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

### Write Documentation

You can never have enough documentation! Please feel free to contribute to any
part of the documentation, such as the official docs, docstrings, or even
on the web in blog posts, articles, and such.

### Submit Feedback

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

## Get Started!

Ready to contribute? Here's how to set up `pyecsago` for local development.

### Prerequisites

* Python >= 3.12
* [uv](https://docs.astral.sh/uv/) (recommended package manager)

### Setup

1. Fork and clone the repository:

    ```bash
    git clone https://github.com/pwnaoj/pyecsago
    cd pyecsago
    ```

2. Install dependencies (including dev tools):

    ```bash
    uv sync
    ```

3. Create a branch for your changes:

    ```bash
    git checkout -b name-of-your-bugfix-or-feature
    ```

4. Make your changes and run the tests:

    ```bash
    uv run pytest --cov --cov-report=term-missing
    ```

5. Commit your changes and open a pull request.

## Coding Standards

* **Python >= 3.12** — use modern type annotations (PEP 585/604):
  `list[T]`, `dict[K, V]`, `X | Y`, `X | None` instead of `List`, `Dict`, `Union`, `Optional`.
* Use `from __future__ import annotations` and `TYPE_CHECKING` for imports only needed by type checkers.
* All new code must include type annotations for parameters and return values.
* Follow [PEP 8](https://peps.python.org/pep-0008/) style conventions.

## Testing

* Tests live in the `tests/` directory and use [pytest](https://docs.pytest.org/).
* Run the full suite with coverage:

    ```bash
    uv run pytest --cov --cov-report=term-missing
    ```

* New features and bug fixes should include corresponding tests.
* Aim for 100% coverage on non-CUDA code paths.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests if appropriate.
2. If the pull request adds functionality, the docs should be updated.
3. The pull request should work for all currently supported versions of Python (3.12+).

## Code of Conduct

Please note that the `pyecsago` project is released with a
Code of Conduct. By contributing to this project you agree to abide by its terms.
