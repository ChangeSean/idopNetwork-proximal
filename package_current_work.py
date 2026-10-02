"""Package only current publication inputs and required source dependencies."""
import hashlib
import json
from pathlib import Path
import zipfile
from release_layout import publication_files

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'deliverables/idop_proximal_complete_work_20261002.zip'


def main():
    audit = json.loads((ROOT / 'results/work_completion_audit_20261001.json').read_text(encoding='utf-8'))
    assert audit['computational_research_workflow_complete']
    document = json.loads((ROOT / 'results/causal_revision_document_audit.json').read_text(encoding='utf-8'))
    assert document['word_sha256'] == hashlib.sha256((ROOT / 'manuscript_SiM.docx').read_bytes()).hexdigest()
    assert document['source_sha256'] == hashlib.sha256((ROOT / 'manuscript.md').read_text(encoding='utf-8').encode()).hexdigest()
    files = publication_files()
    manifest = {'version': '2026-10-02 ten-cohort point estimates with complete uncertainty',
                'primary_allocation': '75:25', 'simulation_offset': 140000,
                'files': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                'patient_inputs_included': False, 'publication_source': 'manuscript.md',
                'word_page_rendering': document['word_page_rendering']}
    target = ROOT / 'results/current_work_manifest_20261001.json'
    target.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    files.append(target)
    OUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT))
    with zipfile.ZipFile(OUT) as archive:
        assert archive.testzip() is None
        for name, expected in manifest['files'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected, name
    result = {'file': OUT.relative_to(ROOT).as_posix(), 'files': len(files), 'size_bytes': OUT.stat().st_size,
              'sha256': hashlib.sha256(OUT.read_bytes()).hexdigest(), 'archive_integrity': True,
              'patient_inputs_included': False}
    (ROOT / 'results/current_work_package_audit_20261001.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (ROOT / 'results/current_revision_manifest.json').write_text(json.dumps({'latest': target.name,
         'audit': 'work_completion_audit_20261001.json', 'manuscript': 'manuscript.md',
         'version': manifest['version']}, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
