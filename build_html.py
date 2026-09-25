"""Execute puzzle.py and export a self-contained illustrated HTML notebook."""

from contextlib import redirect_stdout
from html import escape
from io import StringIO, BytesIO
from pathlib import Path
import base64
import inspect
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jupytext
from markdown_it import MarkdownIt
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
import intro_answers
import IPython.display


def build():
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
    sections, images = [], []
    names = ("four_sides", "six_sides", "triangular", "weighted_die", "two_dice", "single_sample",
                         "ten_samples", "independent_two_variables", "linear_control", "quadratic_control",
                         "two_variables", "additive", "kl_estimate", "kl_k3", "kl_topk",
                         "markov_chain", "markov_unigram", "group_rewards", "reinforce", "reinforce_loo")
    titles = ["Four sides","Six sides","Triangular distribution","Weighted dice from fair coins","Two dice","One sample","Monte Carlo · ten samples","Independent two-variable Monte Carlo","A roughly linear control variate","Five samples, a roughly parabolic function","Five samples, two variables","Five samples, an additive function","k1 KL estimate","k3 KL control variate","Unbiased top-k KL","A two-state Markov chain","A unigram control variate","Additive rewards from one model draw","REINFORCE with a two-dimensional gradient","REINFORCE with leave-one-out"]
    answers = dict(zip(titles, names))
    question_ids = {title: f"puzzle-{i}" for i, title in enumerate(titles, 1)}

    def code(source):
        return highlight(source, PythonLexer(), formatter)

    def capture(*args, **kwargs):
        for number in plt.get_fignums():
            fig = plt.figure(number)
            buffer = BytesIO()
            fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
            encoded = base64.b64encode(buffer.getvalue()).decode()
            images.append(f'<img alt="Exact probability histogram with mean and variance" src="data:image/png;base64,{encoded}">')
            plt.close(fig)

    original_show = plt.show
    original_display = IPython.display.display
    def capture_display(obj):
        if isinstance(obj, IPython.display.Image) and obj.format == "gif":
            encoded = base64.b64encode(obj.data).decode()
            images.append(f'<img alt="Variance of the average of 1 to 20 coin flips" src="data:image/gif;base64,{encoded}">')
            (root / "build").mkdir(exist_ok=True)
            (root / "build" / "coin-variance.gif").write_bytes(obj.data)
        else:
            original_display(obj)
    plt.show = capture
    IPython.display.display = capture_display
    try:
        for cell in notebook.cells:
            if cell.cell_type == "markdown":
                if cell.source.startswith("## "):
                    if sections:
                        sections.append("</section>")
                    heading = cell.source.splitlines()[0][3:]
                    section_id = question_ids.get(heading, heading.lower().replace(" · ", "-").replace(" ", "-").replace(":", "").replace("/", ""))
                    sections.append(f'<section id="{escape(section_id)}" class="puzzle">')
                    sections.append(render_markdown(cell.source))
                    if heading.startswith(("Exercise · ", "Helper · ")):
                        name = heading.split(" · ", 1)[1]
                        if hasattr(intro_answers, name):
                            sections.append(code(inspect.getsource(getattr(intro_answers, name))))
                    if heading not in answers:
                        continue
                    name = answers[heading]
                    answer = inspect.getsource(getattr(intro_answers, name))
                    if name == "ten_samples":
                        answer = inspect.getsource(intro_answers.monte_carlo) + "\n" + answer
                    if name == "kl_k3":
                        answer = inspect.getsource(intro_answers.k3) + "\n" + answer
                    if name == "kl_estimate":
                        answer = inspect.getsource(intro_answers.k1) + "\n" + answer
                    sections.append(code(answer))
                else:
                    sections.append('<section class="intro">' + render_markdown(cell.source))
            elif cell.cell_type == "code":
                output = StringIO()
                images.clear()
                with redirect_stdout(output):
                    exec(compile(cell.source, "puzzle.py", "exec"), namespace)
                if "hide" not in cell.metadata.get("tags", []):
                    sections.append(code(cell.source))
                sections.extend(images)
    finally:
        plt.show = original_show
        IPython.display.display = original_display
        plt.close("all")
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
    navigation = ''.join(f'<a href="#{anchor}">{label}</a>' for anchor, label in
                         (("section-1-random-variables", "Random Variables"),
                          ("section-2-joint-variables", "Joint variables"),
                          ("section-3-elementary-sampling", "Sampling"),
                          ("section-4-monte-carlo", "Monte Carlo"),
                          ("section-5-control-variates", "Control Variates"),
                          ("section-6-example-ab-tests", "A/B Tests"),
                          ("section-7-kl-approximations", "KL Approximations"),
                          ("section-8-markov-chains", "Markov Chains"),
                          ("section-9-group-variance", "Group Variance"),
                          ("section-10-reinforce", "REINFORCE")))
    html = ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>RL Puzzles</title><style>' + css + formatter.get_style_defs('.highlight') +
            '</style><script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>'
            '<body><header><div><strong>RL Puzzles</strong></div><nav>' + navigation +
            '</nav></header><main>' + ''.join(sections) +
            '</main></body></html>')
    output = root / 'build' / 'index.html'
    output.parent.mkdir(exist_ok=True)
    output.write_text(html)
    print(f'Built {output} ({len(html):,} characters; {len(names)} checked plots)')


if __name__ == '__main__':
    build()
