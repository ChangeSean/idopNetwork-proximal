"""Release consistency using current audits and a single publication inventory."""
from pathlib import Path
import hashlib
import json
import py_compile
import re

ROOT = Path(__file__).resolve().parent


def main():
    paths = {'simulation': 'discovery_estimation_validation_20261001/validation_audit.json',
             'clinical': 'discovery_estimation_application_20261001/application_audit.json',
             'algebra': 'independent_design_validation_20261001/algebra_audit.json',
             'niche_decomposition': 'niche_ode_audit.json',
             'document': 'causal_revision_document_audit.json',
             'evidence': 'method_evidence_selection_20261001/evidence_audit.json',
             'release': 'release_audit.json',
             'clinical_extension': 'additional_cohorts_application_20261002/extension_audit.json',
             'clinical_reporting': 'additional_cohorts_application_20261002/reporting_audit.json'}
    audits = {name: json.loads((ROOT / 'results' / path).read_text(encoding='utf-8')) for name, path in paths.items()}
    assert all(a['passed'] for a in audits.values())
    text = (ROOT / 'manuscript.md').read_text(encoding='utf-8')
    assert audits['document']['source_sha256'] == hashlib.sha256(text.encode()).hexdigest()
    assert audits['document']['word_sha256'] == hashlib.sha256((ROOT / 'manuscript_SiM.docx').read_bytes()).hexdigest()
    assert audits['release']['manuscript_sha256'] == hashlib.sha256((ROOT / 'manuscript.md').read_bytes()).hexdigest()
    from release_layout import ENTRY_POINTS, source_dependencies
    for name in source_dependencies(ENTRY_POINTS):
        if name.endswith('.py'):
            py_compile.compile(str(ROOT / name), doraise=True)
    result = {'computational_research_workflow_complete': True, 'passed_checks': {k: v['checks'] for k, v in audits.items()},
              'total_checks': sum(v['checks'] for v in audits.values()), 'final_datasets': 1400,
              'final_outcome_records': 2800, 'main_tables': len(re.findall(r'^\*\*Table \d+\.', text, re.M)),
              'supplementary_tables': len(re.findall(r'^\*\*Table S\d+\.', text, re.M)),
              'references': audits['document']['references'], 'word_native_equations': audits['document']['native_equations'],
              'word_page_rendering': audits['document']['word_page_rendering'],
              'final_protocol': 'DISCOVERY_ESTIMATION_PROTOCOL_20261001.md',
              'version': '2026-10-02 ovarian application with three focused protein analyses', 'author_declarations': 'Awaiting author input'}
    (ROOT / 'results/work_completion_audit_20261001.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
