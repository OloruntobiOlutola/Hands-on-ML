# ...existing code...
import json, re, sys, os

def make_anchor(text):
    a = text.lower()
    a = re.sub(r'<[^>]+>', '', a)                # remove html tags
    a = re.sub(r'\[.*?\]\(.*?\)', '', a)        # remove markdown links
    a = re.sub(r'[^\w\s-]', '', a)              # remove punctuation
    a = re.sub(r'\s+', '-', a).strip('-')       # spaces -> dashes
    return a

def extract_headings(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    headings = []
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'markdown': 
            continue
        lines = cell.get('source', [])
        if isinstance(lines, str): lines = [lines]
        for ln in lines:
            ln = ln.lstrip()
            m = re.match(r'^(#{1,6})\s+(.*)', ln)
            if m:
                level = len(m.group(1))
                text = m.group(2).strip()
                headings.append((level, text))
    return headings

def write_summary(nb_path, headings):
    out = os.path.splitext(nb_path)[0] + '_SUMMARY.md'
    lines = ['# Summary / Table of contents', '']
    for level, text in headings:
        indent = '  ' * (level - 1)
        anchor = make_anchor(text)
        lines.append(f'{indent}- [{text}](#{anchor})')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return out

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python extract_headings.py path/to/notebook.ipynb')
        sys.exit(1)
    nb_path = sys.argv[1]
    headings = extract_headings(nb_path)
    out = write_summary(nb_path, headings)
    print(f'Wrote summary to: {out}')
# ...existing code...