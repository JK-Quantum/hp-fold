"""Chart colors for light and dark surfaces."""

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
)

THEMES = {"light": LIGHT, "dark": DARK}


def get_theme(name: str) -> Theme:
    try:
        return THEMES[name]
    except KeyError:
        raise ValueError(
            f"unknown theme {name!r}; choose from {', '.join(THEMES)}"
        ) from None
