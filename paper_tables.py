"""Table formatting and study labels shared by the paper and figures."""
import math
import re

SCENARIOS = ['rank1_n400_branch', 'rank1_n1000_chain', 'rank1_n1000_modules',
             'rank2_n1000_branch', 'rank2_n1000_branch_strong',
             'rank1_n411_p60_branch', 'rank2_n352_p60_branch']
ROMAN = dict(zip(SCENARIOS, ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']))


def table(caption, headers, rows):
    return caption + '\n\n| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join(['---'] * len(headers)) + ' |\n' + ''.join(
        '| ' + ' | '.join(str(v) for v in row) + ' |\n' for row in rows)


def metric(value, se):
    return f'{value:.3f} ({se:.3f})' if math.isfinite(value) else '—'


def renumber_references(text):
    body, refs = text.split('## References', 1)
    sources = {int(n): desc for n, desc in re.findall(r'^(\d+)\. (.+)$', refs, flags=re.M)}
    order = []
    for match in re.finditer(r'\[(\d+(?:,\s*\d+)*)\]', body):
        for value in map(int, match.group(1).split(',')):
            if value not in order:
                order.append(value)
    assert set(order) == set(sources), (order, sources.keys())
    mapping = {old: new for new, old in enumerate(order, 1)}
    body = re.sub(r'\[(\d+(?:,\s*\d+)*)\]',
                  lambda m: '[' + ', '.join(str(mapping[int(n)]) for n in m.group(1).split(',')) + ']', body)
    return body + '## References\n\n' + '\n'.join(f'{mapping[n]}. {sources[n]}' for n in order) + '\n'
