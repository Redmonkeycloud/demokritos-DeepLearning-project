"""One-shot: regenerate PLAN.html from PLAN.md."""
import markdown, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
md = (ROOT / "PLAN.md").read_text(encoding="utf-8")
html_body = markdown.markdown(md, extensions=["tables", "fenced_code"])

style = """
  body { font-family: system-ui, sans-serif; max-width: 960px; margin: 40px auto;
          padding: 0 24px; line-height: 1.6; color: #1a1a1a; }
  h1 { color: #1a1a2e; border-bottom: 3px solid #e94560; padding-bottom: 8px; }
  h2 { color: #16213e; border-bottom: 1px solid #ccc; padding-bottom: 4px; }
  h3 { color: #0f3460; }
  table { border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 0.9em; }
  th { background: #1a1a2e; color: white; padding: 8px 12px; text-align: left; }
  td { padding: 7px 12px; border-bottom: 1px solid #e0e0e0; }
  tr:nth-child(even) { background: #f8f8f8; }
  code { background: #f0f0f0; padding: 2px 6px; border-radius: 3px;
          font-family: monospace; font-size: 0.88em; }
  pre { background: #1a1a2e; color: #e0e0e0; padding: 16px; border-radius: 6px;
         overflow-x: auto; }
  pre code { background: none; color: inherit; padding: 0; }
  blockquote { border-left: 4px solid #e94560; margin: 0; padding-left: 16px; color: #555; }
"""

html = (
    '<!DOCTYPE html>\n<html lang="el">\n<head>\n'
    '<meta charset="UTF-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
    '<title>Deep Learning SER \u2014 Implementation Plan</title>\n'
    "<style>" + style + "</style>\n"
    "</head>\n<body>\n"
    + html_body
    + "\n</body>\n</html>"
)

(ROOT / "PLAN.html").write_text(html, encoding="utf-8")
print("PLAN.html regenerated.")
print(f"  PLAN.md  : {len(md):,} chars")
print(f"  PLAN.html: {len(html):,} chars")
