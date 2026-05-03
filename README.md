# Physical AI Lab

## Documentation

Run the developer documentation locally with MkDocs:

```bash
uv run --group docs mkdocs serve
```

Build the static documentation site into `site/`:

```bash
uv run --group docs mkdocs build --strict
```

The built site can be opened from `site/index.html` or served as static files.
