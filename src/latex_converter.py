import os
import re
import subprocess
import shutil

class MD2LatexConverter:
    def __init__(self, md_filepath):
        self.md_filepath = md_filepath
        self.base_dir = os.path.dirname(md_filepath)
        self.filename = os.path.basename(md_filepath)
        
        # [新增] 提取 topic 名称（去掉 .md 后缀），用于图片统一命名
        self.topic_name = self.filename.replace('.md', '')
        
        self.tex_filepath = os.path.join(self.base_dir, self.filename.replace('.md', '.tex'))
        self.fig_dir = os.path.join(self.base_dir, 'fig')

    def _escape_latex_text(self, text):
        replacements = {
            '\\': r'\textbackslash{}',
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
        }
        escaped = []
        for ch in text:
            escaped.append(replacements.get(ch, ch))
        return ''.join(escaped)

    def convert_tables(self, match):
        table_text = match.group(0)
        lines = table_text.strip().split('\n')
        if len(lines) < 3: 
            return table_text
        cols = len([c for c in lines[0].split('|') if c.strip()])
        col_format = "@{}" + "".join([">{\\raggedright\\arraybackslash}X" for _ in range(cols)]) + "@{}"
        latex_tb = [
            "\\begin{table}[H]",
            "\\centering",
            "\\small",
            "\\setlength{\\tabcolsep}{4pt}",
            f"\\begin{{tabularx}}{{\\textwidth}}{{{col_format}}}",
            "\\toprule"
        ]
        for line in lines:
            stripped = line.strip()
            sep_cells = [c.strip() for c in stripped.strip('|').split('|')]
            if sep_cells and all(re.fullmatch(r':?-{3,}:?', cell or '') for cell in sep_cells):
                latex_tb.append("\\midrule")
                continue
            cells = [c.strip() for c in line.split('|')]
            if line.strip().startswith('|'): cells = cells[1:]
            if line.strip().endswith('|'): cells = cells[:-1]
            cells = [c.replace('%', '\\%').replace('&', '\\&').replace('_', '\\_').replace('#', '\\#') for c in cells]
            if cells and cells[0].startswith('['):
                cells[0] = '{}' + cells[0]
            row_str = " & ".join(cells) + " \\tabularnewline"
            latex_tb.append(row_str)
        latex_tb.append("\\bottomrule")
        latex_tb.append("\\end{tabularx}")
        latex_tb.append("\\end{table}\n")
        return "\n".join(latex_tb) + "\n"

    def _normalize_heading_key(self, line):
        heading = re.sub(r'^#+\s*', '', line).strip()
        heading = re.sub(r'^\d+(?:\.\d+)*\s*', '', heading).strip()
        return heading.lower()

    def _sanitize_markdown(self, md_text):
        cleaned_lines = []
        last_heading_key = None

        for raw_line in md_text.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()

            if not stripped:
                cleaned_lines.append('')
                continue

            if stripped.startswith('>'):
                continue

            if re.match(r'^\*\(View\s+\d+:.*\)\*$', stripped):
                continue

            if stripped.startswith('```mermaid') or stripped == '```':
                continue

            if 'AI 视觉检查报告' in stripped or '一键修复提示词' in stripped:
                continue

            if re.match(r'^---+$', stripped):
                cleaned_lines.append('')
                continue

            if re.match(r'^#{1,6}\s+', stripped):
                heading_key = self._normalize_heading_key(stripped)
                if heading_key == last_heading_key:
                    continue
                last_heading_key = heading_key
            else:
                last_heading_key = None

            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines) + '\n'

    def _normalize_unicode(self, text):
        replacements = {
            '\u2018': "'",
            '\u2019': "'",
            '\u201c': '"',
            '\u201d': '"',
            '\u2013': '-',
            '\u2014': '--',
            '\u2026': '...',
            '\u00a0': ' ',
            '\u2002': ' ',
            '\u2003': ' ',
            '\u2009': ' ',
            '\u200b': '',
        }
        for src, dst in replacements.items():
            text = text.replace(src, dst)
        return text

    def _escape_currency_dollars(self, text):
        # Preserve real inline math as much as possible, but escape prose currency
        # markers such as "$100K" that otherwise break LaTeX compilation.
        return re.sub(r'(?<!\\)\$(?=\d)', r'\\$', text)

    def _protect_blocks(self, text):
        blocks = []

        def save_block(match):
            blocks.append(match.group(0))
            return f"BLOCKPLACEHOLDER{len(blocks) - 1}BLOCK"

        patterns = [
            r'\\begin\{figure\}.*?\\end\{figure\}',
            r'\\begin\{table\}.*?\\end\{table\}',
        ]

        for pattern in patterns:
            text = re.sub(pattern, save_block, text, flags=re.DOTALL)

        return text, blocks

    def _restore_blocks(self, text, blocks):
        for i, block in enumerate(blocks):
            text = text.replace(f"BLOCKPLACEHOLDER{i}BLOCK", block)
        return text

    def _cleanup_latex_build_files(self):
        for ext in ('.aux', '.log', '.out', '.toc'):
            filepath = self.tex_filepath.replace('.tex', ext)
            if os.path.exists(filepath):
                os.remove(filepath)

    def compile_pdf(self):
        if shutil.which('pdflatex') is None:
            print("    > [Warning] pdflatex not found, skipping PDF compilation.")
            return None

        tex_name = os.path.basename(self.tex_filepath)
        for run_idx in range(2):
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', '-file-line-error', tex_name],
                cwd=self.base_dir,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logs = (result.stdout + '\n' + result.stderr).splitlines()
                tail = '\n'.join(logs[-40:])
                print(f"    > [Error] pdflatex failed on pass {run_idx + 1}:\n{tail}")
                return None

        pdf_filepath = self.tex_filepath.replace('.tex', '.pdf')
        if os.path.exists(pdf_filepath):
            self._cleanup_latex_build_files()
            print(f"    > [Success] PDF successfully generated at: {pdf_filepath}")
            return pdf_filepath

        print("    > [Warning] PDF compilation finished but no PDF was found.")
        return None

    def convert(self):
        if not os.path.exists(self.fig_dir): os.makedirs(self.fig_dir)
        with open(self.md_filepath, 'r', encoding='utf-8') as f:
            md_text = f.read()

        md_text = self._sanitize_markdown(md_text)
        tex_text = re.sub(r'```mermaid\n.*?\n```', '', md_text, flags=re.DOTALL | re.IGNORECASE)

        def build_figure_block(caption, img_rel_path):
            img_name = os.path.basename(img_rel_path)
            old_img_path = os.path.join(self.base_dir, img_name)
            new_img_path = os.path.join(self.fig_dir, img_name)
            if os.path.exists(old_img_path):
                shutil.move(old_img_path, new_img_path)
            caption = self._escape_latex_text(caption.replace('\n', ' ').strip())
            return (f"\\begin{{figure}}[H]\n\\centering\n\\includegraphics[width=0.96\\textwidth,height=0.78\\textheight,keepaspectratio]{{fig/{img_name}}}\n"
                    f"\\caption{{{caption}}}\n\\end{{figure}}\n")

        def bold_caption_image_repl(match):
            caption = match.group(1)
            img_rel_path = match.group(3)
            return build_figure_block(caption, img_rel_path)

        def image_repl(match):
            caption = match.group(1)
            img_rel_path = match.group(2)
            return build_figure_block(caption, img_rel_path)

        tex_text = re.sub(
            r'(?m)^\s*\*\*(Figure:[^\n]*?)\*\*\s*\n\s*!\[(.*?)\]\((.*?)\)\s*$',
            bold_caption_image_repl,
            tex_text,
        )
        tex_text = re.sub(
            r'(?m)^\s*\*\*([^\n]+?)\*\*\s*$',
            lambda m: f"\\paragraph{{{self._escape_latex_text(m.group(1).strip())}}}",
            tex_text,
        )
        tex_text = re.sub(r'!\[(.*?)\]\((.*?)\)', image_repl, tex_text)
        tex_text = re.sub(r'(^\|.*\|\s*\n)+', self.convert_tables, tex_text, flags=re.MULTILINE)
        tex_text = re.sub(r'^# (.*?)$', r'\\title{\1}\n\\maketitle\n', tex_text, flags=re.MULTILINE)
        tex_text = re.sub(r'^## (.*?)$', r'\\section{\1}', tex_text, flags=re.MULTILINE)
        tex_text = re.sub(r'^### (.*?)$', r'\\subsection{\1}', tex_text, flags=re.MULTILINE)
        tex_text = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', tex_text)
        tex_text = re.sub(r'^\s*---+\s*$', '', tex_text, flags=re.MULTILINE)
        tex_text = self._normalize_unicode(tex_text)

        tex_text, blocks = self._protect_blocks(tex_text)
        tex_text = self._escape_currency_dollars(tex_text)
        tex_text = re.sub(r'(?<!\\)&', r'\\&', tex_text)
        tex_text = re.sub(r'(?<!\\)%', r'\\%', tex_text)
        tex_text = re.sub(r'(?<!\\)_', r'\\_', tex_text)
        tex_text = re.sub(r'(?<!\\)#', r'\\#', tex_text)
        tex_text = re.sub(r'(?<!\\)\^', r'\\^{}', tex_text)
        tex_text = self._restore_blocks(tex_text, blocks)
        tex_text = re.sub(r'\n{3,}', '\n\n', tex_text)

        latex_template = f"""\\documentclass[11pt]{{article}}
\\usepackage[T1]{{fontenc}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{graphicx}}
\\usepackage{{float}}
\\usepackage{{geometry}}
\\usepackage[hidelinks]{{hyperref}}
\\usepackage{{booktabs}}
\\usepackage{{tabularx}}
\\usepackage{{array}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\geometry{{a4paper, margin=1in}}

\\begin{{document}}

{tex_text}

\\end{{document}}
"""
        with open(self.tex_filepath, 'w', encoding='utf-8') as f:
            f.write(latex_template)
        
        print(f"    > [Success] LaTeX file successfully generated at: {self.tex_filepath}")
        return self.compile_pdf()
