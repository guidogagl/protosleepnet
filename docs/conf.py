"""Sphinx configuration for the ProtoSleepNet documentation.

The docs are Markdown-first (MyST). Unlike ``physioex`` (whose API docs import
the package), ProtoSleepNet is an *experiment pipeline* of ~90 mostly ``python -m``
CLI modules, so the module reference is produced by **sphinx-autoapi**, which
parses the source statically — the build never imports ``protosleepnet`` /
``physioex`` / ``torch``. That keeps the GitHub-Pages CI light and robust: the
build installs only ``docs/requirements.txt`` (Sphinx + theme + MyST + autoapi),
NOT ``-e .`` — so the heavy core deps never enter the docs environment.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "ProtoSleepNet"
author = "Guido Gagliardi"
copyright = "2026, Guido Gagliardi"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",             # MyST Markdown
    "autoapi.extension",       # static (no-import) API reference over src/
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",  # emits .nojekyll into the build (Pages safe)
    "sphinx_design",           # {grid}, {card}, {tab-set}
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]

root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- MyST --------------------------------------------------------------------

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "deflist",
    "html_image",
    "substitution",
]
# Render bare ```mermaid fences via sphinxcontrib-mermaid (no directive rewrite).
myst_fence_as_directive = ["mermaid"]
myst_heading_anchors = 3

# -- sphinx-autoapi (static module reference) --------------------------------

autoapi_type = "python"
autoapi_dirs = ["../src/protosleepnet"]
autoapi_root = "api"
autoapi_add_toctree_entry = False   # we place ``api/protosleepnet/index`` in the toctree ourselves
autoapi_keep_files = False          # generated stubs are not committed (gitignored)
autoapi_member_order = "groupwise"
# ``undoc-members`` is required: this is a CLI-script pipeline whose members
# rarely carry docstrings, and autoapi skips generating a page for a module with
# no documented members — so without it the whole reference comes out empty.
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
# Skip three figure-emitter scripts that assign a module constant twice — autoapi
# then emits a duplicate (uncategorized, unsuppressable) object description that
# fails the strict ``-W`` build. Their behaviour is documented in the
# reproduction guide; we do not touch the pipeline source.
autoapi_ignore = [
    "*/__pycache__/*",
    "*/figure_reconstruction/combinatorial_ablation.py",
    "*/figure_reconstruction/relevance_signature.py",
    "*/figure_reconstruction/spectral_signature.py",
]

# This repo's docstrings are human prose, not reStructuredText — when autoapi
# renders ~90 module docstrings some contain RST-special constructs (``|IG|``
# reads as a substitution, indented lists, ``*`` emphasis, constants defined
# twice). Those are cosmetic to the auto-generated reference, so we keep the
# strict ``-W`` build for the hand-written Guide/content while scoping out the
# autoapi + docstring-RST categories. MyST/toctree/xref problems in our own
# pages (categories ``myst.*`` / ``toc.*`` / ``ref.*``) are NOT suppressed and
# still fail the build.
suppress_warnings = [
    "autoapi",   # static import-resolution notes
    "docutils",  # prose docstrings rendered as RST
    "ref.python",
    "epub.duplicated_definition",
]

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "physioex": ("https://guidogagl.github.io/physioex", None),
}

# -- Mermaid -----------------------------------------------------------------

mermaid_version = "11.4.0"

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "ProtoSleepNet"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/guidogagl/protosleepnet",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "Live demo",
            "url": "https://protosleepnet-demo.pages.dev",
            "icon": "fa-solid fa-play",
        },
        {
            "name": "physioex",
            "url": "https://github.com/guidogagl/physioex",
            "icon": "fa-solid fa-cube",
        },
    ],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "use_edit_page_button": False,
}
html_context = {
    "default_mode": "light",
    "github_user": "guidogagl",
    "github_repo": "protosleepnet",
    "github_version": "main",
}
