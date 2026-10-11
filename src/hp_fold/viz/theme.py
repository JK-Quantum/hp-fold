"""Chart colors for light and dark surfaces."""

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Theme:
    name: str
    surface: str
    ink: str
    ink_secondary: str
    ink_muted: str
    grid: str
    axis: str
    accent: str
    neutral: str
    alert: str
    series: tuple[str, ...]


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    ink_muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    accent="#2a78d6",
    neutral="#898781",
    alert="#eb6834",
    series=(
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
        "#008300",
        "#6250d6",
        "#e34948",
    ),
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    ink="#f0efec",
    ink_secondary="#c3c2b7",
    ink_muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    accent="#3987e5",
    neutral="#898781",
    alert="#d95926",
    series=(
        "#3987e5",
        "#d95926",
        "#199e70",
        "#c98500",
        "#d55181",
        "#008300",
        "#9085e9",
        "#e66767",
    ),
)

THEMES = {"light": LIGHT, "dark": DARK}


def get_theme(name: str) -> Theme:
    try:
        return THEMES[name]
    except KeyError:
        raise ValueError(
            f"unknown theme {name!r}; choose from {', '.join(THEMES)}"
        ) from None


_SLOT_OF = {
    "random": 0,
    "annealing": 1,
    "qaoa": 2,
    "qaoa_cvar": 3,
    "vqe": 4,
    "qaoa_noisy": 6,
}
_KNOWN_LONGEST_FIRST = sorted(_SLOT_OF, key=len, reverse=True)


def _normalize(algorithm: str) -> str:
    return algorithm.strip().lower().replace("-", "_").replace(" ", "_")


def algorithm_colors(theme: Theme, algorithms: Iterable[str]) -> dict[str, str]:
    """One color per algorithm name.

    Known algorithms keep a fixed slot, others take a free one.
    """
    names = list(dict.fromkeys(algorithms))
    slots: dict[str, int] = {}
    for name in names:
        normalized = _normalize(name)
        known = next(
            (k for k in _KNOWN_LONGEST_FIRST if normalized.startswith(k)),
            None,
        )
        if known is not None:
            slots[name] = _SLOT_OF[known]
    free = (i for i in range(len(theme.series)) if i not in set(slots.values()))
    for name in names:
        if name not in slots:
            try:
                slots[name] = next(free)
            except StopIteration:
                raise ValueError(
                    f"at most {len(theme.series)} distinct algorithms can be colored"
                ) from None
    return {name: theme.series[slot] for name, slot in slots.items()}
