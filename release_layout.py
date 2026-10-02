"""Explicit publication inputs and transitive local Python dependencies."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULT_DIRS = ('joint_readout_validation_20261001', 'conditional_design_validation_20261001',
               'concentrated_bridge_validation_20261001', 'conditional_design_application_20261001',
               'independent_design_validation_20261001', 'discovery_estimation_validation_20261001',
               'discovery_estimation_application_20261001', 'additional_cohorts_application_20261002')
ENTRY_POINTS = ('workflow.py', 'build_current_manuscript.py', 'build_docx.py', 'audit_causal_revision.py',
                'audit_current_work.py', 'audit_release.py', 'package_current_work.py', 'release_layout.py',
                'select_method_evidence.py', 'make_independent_design_figures.py', 'check_independent_design.py',
                'validate_discovery_estimation.py', 'summarize_discovery_estimation.py',
                'run_discovery_estimation_application.py', 'run_independent_structured_cases.py',
                'audit_discovery_estimation_application.py', 'fetch_tcga.py',
                'validate_joint_readout_bridge.py', 'validate_joint_comparators.py', 'summarize_joint_readout.py',
                'clinical_reporting.py', 'audit_clinical_reporting.py', 'run_remaining_clinical.py',
                'summarize_additional_clinical.py')
DOCUMENTS = ('manuscript.md', 'manuscript_SiM.docx', 'README.md', 'REPRODUCIBILITY.md', 'requirements.txt',
             'requirements-documents.txt', 'DISCOVERY_ESTIMATION_PROTOCOL_20261001.md',
             'clinical_case_boundaries_20261001.json', 'THIRD_PARTY_NOTICES.md', 'third_party/LICENSE.idopnetwork')
FIGURES = ('fig1_schematic', 'fig2_fits', 'fig3_causal', 'fig4_simulation', 'fig5_cohorts',
           'fig6_application', 'fig7_clinical_cases', 'figS1_full_network', 'figS2_network',
           'figS3_report_rates', 'figS4_proxy_construction')


def source_dependencies(seeds):
    pending, found = list(seeds), set()
    while pending:
        name = pending.pop()
        if name in found:
            continue
        path = ROOT / name
        assert path.is_file(), name
        found.add(name)
        if path.suffix != '.py':
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            modules = ([node.module] if isinstance(node, ast.ImportFrom) and node.module else
                       [n.name for n in node.names] if isinstance(node, ast.Import) else [])
            for module in modules:
                local = module.split('.')[0] + '.py'
                if (ROOT / local).is_file():
                    pending.append(local)
    return found


def publication_files():
    files = {ROOT / name for name in DOCUMENTS}
    files.add(ROOT / 'results/additional_cohorts_application_20261002/EXTENSION_PROTOCOL.md')
    seeds = set(ENTRY_POINTS)
    for folder in RESULT_DIRS:
        for path in (ROOT / 'results' / folder).iterdir():
            if path.suffix not in ('.csv', '.json') or not path.is_file():
                continue
            files.add(path)
            if path.suffix == '.json':
                data = json.loads(path.read_text(encoding='utf-8'))
                for name in data.get('source_sha256', {}):
                    if (ROOT / name).is_file():
                        if name.endswith('.py'):
                            seeds.add(name)
                        else:
                            files.add(ROOT / name)
    files.update(ROOT / name for name in source_dependencies(seeds))
    for name in FIGURES:
        for extension in ('.pdf', '.png'):
            path = ROOT / 'figures' / (name + extension)
            if path.is_file():
                files.add(path)
    for name in ('all_paired_comparisons.csv', 'comparison_overview.csv', 'evidence_audit.json'):
        files.add(ROOT / 'results' / 'method_evidence_selection_20261001' / name)
    for name in ('release_audit.json', 'causal_revision_document_audit.json', 'work_completion_audit_20261001.json'):
        files.add(ROOT / 'results' / name)
    for path in files:
        assert path.is_file(), path
        assert 'data' not in path.relative_to(ROOT).parts
        assert not path.name.startswith(('update_manuscript_', 'extend_manuscript_')), path
    return sorted(files)
