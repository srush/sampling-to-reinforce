"""Execute puzzle.py and export a self-contained illustrated HTML notebook."""

from contextlib import redirect_stdout
from html import escape
from io import StringIO
from pathlib import Path
import inspect
import os
import re
import sys
from time import perf_counter

os.environ["RL_PUZZLES_EMBED_PLOTLY_JS"] = "0"

import jupytext
from markdown_it import MarkdownIt
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
import intro_answers
import IPython.display


def build(profile=False):
    root = Path(__file__).resolve().parent
    notebook = jupytext.read(root / "puzzle.py")
    formatter = HtmlFormatter()
    markdown = MarkdownIt("commonmark")
    def render_markdown(source):
        # Keep LaTeX intact instead of treating backslashes as Markdown escapes.
        chunks = re.split(r"(\$\$[\s\S]*?\$\$)", source)
        return ''.join('<div class="math">\\[' + escape(chunk[2:-2].strip()) + '\\]</div>'
                       if chunk.startswith('$$') else markdown.render(chunk) for chunk in chunks)
    namespace = {"__name__": "__main__"}
    timings = []
    sections, images = [], []
    section_nav = []
    section_open = False
    names = ("four_sides", "six_sides", "triangular", "weighted_die", "two_triangles", "monte_carlo",
                         "monte_carlo", "independent_two_variables", "linear_control", "quadratic_control",
                         "two_variables", "additive", "k1", "k3", "kl_topk",
                         "markov_chain", "markov_unigram", "group_rewards", "reinforce", "reinforce_loo")
    titles = ["Four sides","Six sides","Triangular distribution","Weighted dice from fair coins","Two dice","One sample","Monte Carlo · ten samples","Independent two-variable Monte Carlo","A roughly linear control variate","Five samples, a roughly parabolic function","Five samples, two variables","Five samples, an additive function","k1 KL estimate","k3 KL control variate","Unbiased top-k KL","A two-state Markov chain","A unigram control variate","Additive rewards from one model draw","REINFORCE with a two-dimensional gradient","REINFORCE with leave-one-out"]
    titles[4] = "Two triangles"
    answers = dict(zip(titles, names))
    # These estimators are defined in the visible notebook cells.
    answers.pop("k1 KL estimate")
    answers.pop("k3 KL control variate")
    question_ids = {title: f"puzzle-{i}" for i, title in enumerate(titles, 1)}
    # Keep later puzzle IDs stable while removing this example from the page.
    answers.pop("Five samples, an additive function")
    question_ids.pop("Five samples, an additive function")

    def code(source):
        return highlight(source, PythonLexer(), formatter)

    original_display = IPython.display.display
    from plotly.offline import get_plotlyjs
    plotly_bundle = get_plotlyjs()
    def capture_display(obj):
        if isinstance(obj, IPython.display.HTML):
            images.append(obj.data)
        else:
            original_display(obj)
    IPython.display.display = capture_display
    try:
        for cell_number, cell in enumerate(notebook.cells, 1):
            started = perf_counter()
            if cell.cell_type == "markdown":
                if cell.source.startswith("## "):
                    if section_open:
                        sections.append("</section>")
                    heading = cell.source.splitlines()[0][3:]
                    section_id = question_ids.get(heading, heading.lower().replace(" · ", "-").replace(" ", "-").replace(":", "").replace("/", ""))
                    sections.append(f'<section id="{escape(section_id)}" class="puzzle">')
                    if heading.startswith("Section "):
                        section_nav.append((section_id, heading.split(" · ", 1)[-1]))
                    section_open = True
                    sections.append(render_markdown(cell.source))
                    if heading.startswith(("Exercise · ", "Helper · ")):
                        name = heading.split(" · ", 1)[1]
                        if hasattr(intro_answers, name):
                            sections.append(code(inspect.getsource(getattr(intro_answers, name))))
                    if heading not in answers:
                        continue
                    name = answers[heading]
                    answer = inspect.getsource(getattr(intro_answers, name))
                    sections.append(code(answer))
                else:
                    if not section_open:
                        sections.append('<section id="introduction" class="intro">')
                        section_open = True
                    sections.append(render_markdown(cell.source))
                    if cell.source.startswith(("### Exercise · ", "### Helper · ")):
                        name = cell.source.splitlines()[0].split(" · ", 1)[1]
                        if hasattr(intro_answers, name):
                            sections.append(code(inspect.getsource(getattr(intro_answers, name))))
            elif cell.cell_type == "code":
                output = StringIO()
                images.clear()
                with redirect_stdout(output):
                    try:
                        exec(compile(cell.source, "puzzle.py", "exec"), namespace)
                    except Exception as exc:
                        first_line = cell.source.strip().splitlines()[0] if cell.source.strip() else "(empty)"
                        raise RuntimeError(f"Notebook cell {cell_number} failed: {first_line}") from exc
                if "hide" not in cell.metadata.get("tags", []):
                    sections.append(code(cell.source))
                    if output.getvalue():
                        sections.append('<pre class="output">' + escape(output.getvalue()) + '</pre>')
                sections.extend(images)
            if profile:
                label = cell.source.splitlines()[0].strip() if cell.source.strip() else "(empty)"
                timings.append((perf_counter() - started, cell_number, label[:90]))
    finally:
        IPython.display.display = original_display
    if section_open:
        sections.append("</section>")
    css = """
    :root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#faf9f5;color:#24353c;
    font:16px/1.65 system-ui,-apple-system,sans-serif}header{padding:22px 6vw;border-bottom:1px solid #deded6;
    display:flex;justify-content:space-between;gap:20px}header span{font-size:13px;color:#6c787c}
    nav{display:flex;gap:14px;flex-wrap:wrap}a{color:#3e7486;text-decoration:none}main{max-width:920px;margin:auto;padding:35px 28px 80px}
    h1{font-size:42px;letter-spacing:-1.4px;line-height:1.15}h2{font-size:26px;letter-spacing:-.5px}
    .intro{margin-bottom:44px}.puzzle{border-top:1px solid #d8dcd9;padding:24px 0 32px;scroll-margin-top:20px}
    .math{overflow-x:auto;margin:20px 0 26px}
    p{max-width:78ch}code{font-size:.9em}p code{background:#eaf0ed;border-radius:4px;padding:2px 5px}
    .highlight{background:#eef1ed!important;padding:14px 18px;border-radius:9px;overflow:auto}
    pre{margin:0;font:13px/1.6 ui-monospace,Menlo,monospace}details{margin:16px 0}
    summary{cursor:pointer;font-size:13px;color:#52696b;font-weight:600;margin-bottom:7px}
    .passed{font-size:13px;color:#367052;background:#eaf1e8;border-radius:6px;padding:8px 12px}
    img{display:block;width:100%;height:auto;margin:20px 0 0}footer{color:#708084;font-size:12px;padding-top:25px}
    @media(max-width:600px){header{display:block}nav{margin-top:12px}main{padding:20px 16px}h1{font-size:32px}}
    """
    navigation = ''.join(f'<a href="#{escape(anchor)}">{escape(label)}</a>'
                         for anchor, label in section_nav)
    html = ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>RL Puzzles</title><style>' + css + formatter.get_style_defs('.highlight') +
            '</style><script>' + plotly_bundle + '</script>'
            '<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>'
            '<body><header><div><strong>RL Puzzles</strong></div><nav>' + navigation +
            '</nav></header><main>' + ''.join(sections) +
            '</main></body></html>')
    output = root / 'build' / 'index.html'
    output.parent.mkdir(exist_ok=True)
    output.write_text(html)
    print(f'Built {output} ({len(html):,} characters; {len(names)} checked plots)')
    if profile:
        for elapsed, number, label in sorted(timings, reverse=True)[:15]:
            print(f'{elapsed:6.2f}s  cell {number:3d}  {label}')


if __name__ == '__main__':
    build(profile='--profile' in sys.argv)
