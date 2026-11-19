# Contributing

Bug fixes, docs and focused improvements are welcome. For a larger change,
open an issue first so we can discuss the approach.

Fork the repository, make one focused change and open a pull request against
`main`. Explain what changes, why it matters and how you tested it.

```sh
python -m pip install ".[dev,test]"
python -m pytest
```

Follow the Ruff configuration in pyproject.toml and include a regression test for behavior changes.

The automated checks run before review. Gabriel reviews each contribution and
decides whether to merge it or request changes.
