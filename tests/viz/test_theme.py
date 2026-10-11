import pytest

from hp_fold.viz.theme import THEMES, get_theme


def luminance(hex_color: str) -> float:
    """WCAG relative luminance of '#rrggbb'."""
    channels = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [
        c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels
    ]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_light_and_dark_themes_exist():
    assert set(THEMES) == {"light", "dark"}
    assert get_theme("light").name == "light"
    assert get_theme("dark").name == "dark"


def test_unknown_theme_is_rejected_and_the_choices_are_listed():
    with pytest.raises(ValueError, match="light"):
        get_theme("sepia")


def test_the_dark_surface_is_darker_than_the_light_one():
    assert luminance(get_theme("dark").surface) < luminance(get_theme("light").surface)


@pytest.mark.parametrize("name", ["light", "dark"])
@pytest.mark.parametrize("role", ["accent", "neutral", "alert"])
def test_mark_colors_have_at_least_3_to_1_contrast_with_their_surface(name, role):
    theme = get_theme(name)
    assert contrast(getattr(theme, role), theme.surface) >= 3.0


@pytest.mark.parametrize("name", ["light", "dark"])
def test_primary_and_secondary_text_are_readable(name):
    theme = get_theme(name)
    assert contrast(theme.ink, theme.surface) >= 7.0
    assert contrast(theme.ink_secondary, theme.surface) >= 4.5


@pytest.mark.parametrize("name", ["light", "dark"])
def test_the_three_mark_roles_are_distinct(name):
    theme = get_theme(name)
    assert len({theme.accent, theme.neutral, theme.alert}) == 3
