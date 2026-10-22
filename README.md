# python-coding-style

A Claude Code plugin with one skill for writing scientific Python: minimal code that fails loudly. It works on its
own in any Python project and needs no other plugin.

| Part | What it does |
| --- | --- |
| `skills/python-coding-style/` | Conventions for scientific Python: think before coding, minimal code and surgical changes, no `try`/`except`, asserts where numpy, pandas and compiled libraries fail silently, one source of truth and one data object, no magic thresholds, tests on synthetic data plus quick-look checks on real data, type hints and scanpy-style docstrings. |

## Use

The skill loads on its own whenever Claude writes, reviews, edits or debugs Python, including notebooks. To load it
explicitly, run `/python-coding-style:python-coding-style`.

On claude.ai or in the Claude desktop app, zip the skill folder and upload it under **Customize > Skills**
(**+** > **Create skill** > **Upload a skill**):

```bash
cd skills && zip -r python-coding-style.zip python-coding-style
```

## Install

Once per machine, from the [quadbio marketplace](https://github.com/quadbio/claude-plugins):

```bash
claude plugin marketplace add quadbio/claude-plugins
claude plugin install python-coding-style@quadbio
```

## Develop

```bash
claude plugin validate .
claude --plugin-dir .        # try the skill from this checkout
```

## Release

Bump `version` in `.claude-plugin/plugin.json` and merge: the marketplace follows `main`, and users update when the
version changes.

## License

MIT
