"""Rewrite README.md in place for PyPI, which cannot resolve repository-relative paths.

    python .github/pypi-readme.py

Relative images (`src=`, `srcset=`, `![](...)`) become raw.githubusercontent.com URLs and
relative links (`href=`, `[](...)`) github.com/.../blob/ URLs, both on `main` (the current
files, also for older releases).
The README in the repository stays relative so GitHub renders it on every branch.
Run by release.yml before maturin builds the sdist and wheels.
"""

import re
from pathlib import Path

REPO = 'plainerman/better-allegro-buddy'
REF = 'main'


def is_relative(target: str) -> bool:
    return not re.match(r'([a-z][a-z0-9+.-]*:|#|/)', target, re.IGNORECASE)


def main() -> None:
    raw = f'https://raw.githubusercontent.com/{REPO}/{REF}/'
    blob = f'https://github.com/{REPO}/blob/{REF}/'

    def absolute(base: str, target: str) -> str:
        return base + target.removeprefix('./') if is_relative(target) else target

    path = Path('README.md')
    text = path.read_text(encoding='utf-8')
    text = re.sub(
        r'\b(src|srcset)="([^"]+)"',
        lambda m: f'{m[1]}="{absolute(raw, m[2])}"',
        text,
    )
    text = re.sub(r'\bhref="([^"]+)"', lambda m: f'href="{absolute(blob, m[1])}"', text)
    text = re.sub(
        r'(!?)\[([^\]]*)\]\(([^)\s]+)\)',
        lambda m: f'{m[1]}[{m[2]}]({absolute(raw if m[1] else blob, m[3])})',
        text,
    )
    path.write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
