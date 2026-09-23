"""Fija las rutas de trabajo de Ultralytics al repositorio.

Ultralytics guarda `datasets_dir` / `runs_dir` / `weights_dir` en un archivo de
configuracion **global del usuario** (`~/.config/Ultralytics/settings.json`),
no en el proyecto. Eso significa que:

- las rutas que uses quedan grabadas fuera del repo y sobreviven a borrarlo;
- un `project=` relativo se resuelve contra ese directorio global, no contra el
  cwd, asi que los artefactos terminan en un lugar que nadie espera;
- otra persona que clone el repo obtiene rutas distintas y no encuentra los
  archivos que el informe cita.

Caso real que motiva este modulo (auditoria S8): la configuracion global quedo
apuntando a un worktree de git que ya no existe, y una evaluacion lanzada con
`project="reports/paper/val_runs"` escribio en
`.../.claude/worktrees/yolo-macro-detect-hardening/runs/detect/reports/paper/val_runs/`.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def pin_ultralytics_paths(repo_root: Path | None = None) -> dict[str, str]:
    """Ancla las rutas de Ultralytics al repositorio. Idempotente.

    Returns:
        Las rutas efectivas, para poder registrarlas junto a los resultados.
    """
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    rutas = {
        "datasets_dir": str(root / "datasets"),
        "runs_dir": str(root / "runs"),
        "weights_dir": str(root / "weights"),
    }

    try:
        from ultralytics import settings
    except ImportError:  # sin ultralytics no hay nada que fijar
        return rutas

    if {k: settings.get(k) for k in rutas} != rutas:
        settings.update(rutas)
    return rutas
