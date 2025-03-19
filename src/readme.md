# How to generate the document

run

```bash
python3 scripts/gen-md.py
```


# How to add a paper

Simply add a diction in the `scripts/metadata.json`.

The allowed keywords for links: `paper` (link to the paper), `github` (link to github), `huggingface` (link to hugging face), `webpage` (link to other web page).

If you are unsure about the `h2`, `h3`, `h4` allowed, please refer to `scripts/structure.yml`.


# How to modify the structure

Modify the structure in `scripts/structure.yml`.
