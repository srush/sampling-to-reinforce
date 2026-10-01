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
    local_definitions = {name for cell in notebook.cells if cell.cell_type == "code"
                         for name in re.findall(r"^def (\w+)\(", cell.source, re.MULTILINE)}
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
                         "two_variables", "additive", "k1", "kl_k3", "kl_topk",
                         "markov_chain", "markov_unigram", "group_rewards", "reinforce", "reinforce_loo")
    titles = ["Four sides","Six sides","Triangular distribution","Weighted dice from fair coins","Two dice","One sample","Monte Carlo · ten samples","Independent two-variable Monte Carlo","A roughly linear control variate","Five samples, a roughly parabolic function","Five samples, two variables","Five samples, an additive function","k1 KL estimate","k3 KL control variate","Unbiased top-k KL","A two-state Markov chain","A unigram control variate","Additive rewards from one model draw","REINFORCE with a two-dimensional gradient","REINFORCE with leave-one-out"]
    titles[4] = "Two triangles"
    answers = dict(zip(titles, names))
    # These estimators are defined in the visible notebook cells.
    answers.pop("k1 KL estimate")
    answers.pop("k3 KL control variate")
    answers.pop("REINFORCE with leave-one-out")
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
                        if name not in local_definitions and hasattr(intro_answers, name):
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
                        if name not in local_definitions and hasattr(intro_answers, name):
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
    /* Article styling follows srush/lean-transformer's Tufte/Verso theme. */
    :root{color-scheme:light;font-size:15px}*{box-sizing:border-box}
    body{margin:0;background:#fff;color:#404444;
      font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;
      font-size:1.2rem;line-height:1.6}
    main{width:58rem;max-width:100%;margin:auto;padding:0 2rem 8rem}
    header{display:none}
    a{color:#305c78;text-decoration:underline;text-underline-offset:.15em;text-decoration-thickness:1px}
    a:hover{background:#edf2f5}a:focus-visible,summary:focus-visible{outline:2px solid #305c78;outline-offset:3px}
    h1,h2,h3{font-weight:400;line-height:1.15}
    h1{font-size:3.2rem;margin:3rem 0 1.5rem}
    h2{font-size:2.2rem;margin:3rem 0 1rem}
    h3{font-size:1.7rem;margin:2rem 0 1rem}
    .intro{margin-bottom:1.8rem}.puzzle{padding:1rem 0;scroll-margin-top:1.5rem}
    p{margin:1.4rem 0}ul,ol{padding-left:2rem}li{margin:.6rem 0}
    .math{overflow-x:auto;margin:1.5rem 0;font-size:1.2rem}
    code,pre{font-family:Consolas,'Liberation Mono',Menlo,monospace}
    :not(pre)>code{font-size:.8em;background:#f4f4f4;padding:.1em .2em}
    .highlight{background:#f8f8f8!important;border:1px solid #e5e5e5;
      padding:1rem 1.2rem;margin:1.4rem 0;overflow-x:auto;border-radius:0}
    pre{margin:0;font-size:1rem;line-height:1.5;tab-size:4}
    pre.output{color:#666;font-size:.9rem;white-space:pre-wrap;overflow-wrap:anywhere;margin:1rem 0 1.5rem}
    details{margin:1.4rem 0}summary{cursor:pointer;color:#666;font-size:1.1rem}
    .passed{font-size:1rem;color:#367052;padding:.5rem 0}
    img{display:block;max-width:100%;height:auto;margin:1.4rem 0}
    blockquote{border-left:4px solid #ccc;margin:2rem 0;padding:0 2rem}
    footer{color:#666;font-size:1rem;padding-top:2rem}
    @media(max-width:768px){main{padding:0 1.2rem 4rem}h1{font-size:2.6rem}
      h2{font-size:2rem}.highlight{padding:.8rem}}
    @media print{main{width:100%;padding:0}.highlight{overflow:visible}pre{white-space:pre-wrap}}
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
