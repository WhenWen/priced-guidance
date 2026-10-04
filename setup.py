from pathlib import Path
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class build_py(_build_py):
    """Copy canonical submissions into the generated wheel build tree."""

    def run(self) -> None:
        super().run()
        project = Path(__file__).resolve().parent
        destination = Path(self.build_lib) / "tech_tree_arena" / "data" / "submission_templates"
        for name, source in _templates(project).items():
            target = destination / name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(
                source,
                target,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
            )

    def get_outputs(self, include_bytecode: int = 1) -> list[str]:
        outputs = list(super().get_outputs(include_bytecode))
        project = Path(__file__).resolve().parent
        destination = Path(self.build_lib) / "tech_tree_arena" / "data" / "submission_templates"
        for name, source in _templates(project).items():
            outputs.extend(
                str(destination / name / path.relative_to(source))
                for path in source.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
            )
        return outputs


def _templates(project: Path) -> dict[str, Path]:
    submissions = project / "submissions"
    return {
        "reference_pair": submissions / "reference_pair",
        "minimal_pair": submissions / "examples" / "minimal_pair",
    }


setup(cmdclass={"build_py": build_py})
