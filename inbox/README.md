# Inbox

For sources that prohibit automated collection (Reuters), a licensed reader saves the article
page here (browser: *Save page as → Webpage, HTML only*), then runs:

    python -m mqm import-inbox

The page's canonical URL identifies the source and article; it then goes through the same
extraction, LLM span selection and deterministic verification as crawled pages, and its raw HTML
is kept as an audit snapshot. Processed files are renamed to `*.done`.
