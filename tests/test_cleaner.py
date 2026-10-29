import pytest

from jklu_compass.ingestion.cleaner import CleaningConfig, DocumentCleaner

def test_cleaner_removes_jklu_template_navigation() -> None:
    """Recognized template navigation and its footer should be removed."""
    cleaner = DocumentCleaner()

    text = """# Placements

## Placement Support that Sets You Apart

JKLU provides placement preparation and career support.

## Connect with Us

Mr. Rahul Dev Kahera
Phone: +91 9760404458
Email: rahul.kahera@jklu.edu.in

- ### [About Us](https://jklu.edu.in/placements#)
- [About JKLU](https://jklu.edu.in/about-us/about-jklu)
- ### [Quick Links](https://jklu.edu.in/placements#)

#### Other Links

- [Privacy Policy](https://jklu.edu.in/privacy-policy)
"""

    result = cleaner.clean(text)

    assert "# Placements" in result
    assert "placement preparation and career support" in result
    assert "Mr. Rahul Dev Kahera" in result
    assert "+91 9760404458" in result
    assert "rahul.kahera@jklu.edu.in" in result
    assert "About Us" not in result
    assert "Privacy Policy" not in result


def test_cleaner_preserves_documents_without_jklu_navigation() -> None:
    """Ordinary documents should not be truncated."""
    cleaner = DocumentCleaner()

    text = """# Academic Policies

## Other Links

Students can find additional academic resources in this section.
"""

    result = cleaner.clean(text)

    assert result.rstrip() == text.rstrip()
    assert "# Academic Policies" in result
    assert "## Other Links" in result
    assert "Students can find additional academic resources in this section." in result
def test_cleaner_can_disable_jklu_navigation_removal() -> None:
    """The navigation-removal behavior should be configurable."""
    cleaner = DocumentCleaner(
        CleaningConfig(remove_jklu_navigation=False)
    )

    text = """# Placements

Useful placement information.

- ### [About Us](https://jklu.edu.in/placements#)

#### Other Links

- [Privacy Policy](https://jklu.edu.in/privacy-policy)
"""

    result = cleaner.clean(text)

    assert "About Us" in result
    assert "Privacy Policy" in result
