import nox

nox.options.sessions = ["lint", "typecheck"]


@nox.session
def lint(session: nox.Session) -> None:
    """Check code with ruff."""
    session.run("uv", "run", "ruff", "check", ".")


@nox.session
def format(session: nox.Session) -> None:
    """Format code with ruff."""
    session.run("uv", "run", "ruff", "format", ".")


@nox.session
def typecheck(session: nox.Session) -> None:
    """Type check with pyright."""
    session.run("uv", "run", "pyright")


@nox.session
def fix(session: nox.Session) -> None:
    """Auto-fix all issues."""
    session.run("uv", "run", "ruff", "check", "--fix", ".")
    session.run("uv", "run", "ruff", "format", ".")


@nox.session
def pre_commit(session: nox.Session) -> None:
    """Run before commit."""
    session.run("uv", "run", "ruff", "check", "--fix", ".")
    session.run("uv", "run", "ruff", "format", ".")
    session.run("uv", "run", "pyright")


@nox.session
def ci(session: nox.Session) -> None:
    """Full CI pipeline."""
    session.run("uv", "run", "ruff", "check", ".")
    session.run("uv", "run", "ruff", "format", "--check", ".")
    session.run("uv", "run", "pyright")
