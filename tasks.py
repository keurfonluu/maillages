import glob
import os
import shutil
import tarfile

from invoke.tasks import task


@task
def build(c):
    shutil.rmtree("dist", ignore_errors=True)
    c.run("python -m build --sdist --wheel .")


@task
def html(c):
    c.run("sphinx-build -b html doc/source doc/build")


@task
def tag(c):
    import maillages
    
    c.run(f"git tag v{maillages.__version__}")
    c.run("git push --tags")


@task
def clean(c, bytecode=False):
    patterns = [
        "build",
        "dist",
        "maillages.egg-info",
        "doc/build",
        "doc/source/examples",
    ]
    patterns += glob.glob("**/.coverage*", recursive=True)

    if bytecode:
        patterns += glob.glob("**/*.pyc", recursive=True)
        patterns += glob.glob("**/__pycache__", recursive=True)

    for pattern in patterns:
        if os.path.isfile(pattern):
            os.remove(pattern)
        else:
            shutil.rmtree(pattern, ignore_errors=True)


@task
def ruff(c):
    c.run("ruff check --fix maillages tests")
    c.run("ruff format --target-version py39 --line-length 88 maillages tests")


@task
def tar(c):
    patterns = [
        "__pycache__",
    ]

    def filter(filename):
        for pattern in patterns:
            if filename.name.endswith(pattern):
                return None

        return filename

    with tarfile.open("maillages.tar.gz", "w:gz") as tf:
        tf.add("maillages", arcname="maillages/maillages", filter=filter)
        tf.add("pyproject.toml", arcname="maillages/pyproject.toml")


@task
def test(c, cov=False, html=False):
    command = ["python -m pytest"]

    if cov:
        command += ["--cov", "maillages", "--cov-report", "term"]

        if html:
            command += ["--cov-report", "html"]

    command += ["tests"]

    c.run(" ".join(command))
