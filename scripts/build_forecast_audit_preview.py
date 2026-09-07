#!/usr/bin/env python3
"""Assemble verified forecast-audit research in an isolated local preview.

No operational issue, git operation or external publication is performed. The
archived official weekly forecasts remain byte-for-byte equivalent as objects.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from regime_lab.contract_v5 import validate_v5_payload
from regime_lab.dashboard_split import build_dashboard_split, build_history_chunks
from regime_lab.publication_contract import rewrite_index_asset_versions
from scripts.package_public_demo import STATIC_ALLOWLIST


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n').encode()


def require_local_output(output: Path) -> None:
    selected = output.resolve()
    allowed = (ROOT / 'build' / 'forecast-audit-improvements').resolve()
    if selected != allowed and allowed not in selected.parents:
        raise ValueError('audit preview output must stay under build/forecast-audit-improvements')
    if output.is_symlink() and (ROOT / 'publication').resolve() in selected.parents:
        raise ValueError('preview cannot resolve into publication')


def build_preview(source: Path, research_files: dict[str,Path], output: Path) -> dict:
    require_local_output(output)
    source_raw=source.read_bytes()
    original=json.loads(source_raw)
    payload=deepcopy(original)
    bindings={}
    research_bytes={}
    companion_bytes={}
    artifact_labels={
        'forecast_research':'모델·경로 전체 연구 결과 JSON',
        'calibration_audit':'확률 보정 비교 전체 결과 JSON',
        'forecast_information':'추가 정보 비교 전체 결과 JSON',
    }
    for key,path in research_files.items():
        raw=path.read_bytes()
        research_bytes[key]=raw
        document=json.loads(raw)
        if not isinstance(document,dict): raise ValueError(f'{key} must be a JSON object')
        if key in artifact_labels:
            document['artifacts']=[*document.get('artifacts',[]),
                {'label':artifact_labels[key],'url':f'./data/{key}.json'}]
        if key == 'forecast_information':
            for name,label in (('summary.json','추가 데이터·현재 피처 전체 결과'),
                               ('preview.html','추가 데이터 상세 미리보기')):
                companion=path.parent/name
                if companion.is_file():
                    destination_name=f'data/new-information-{name}'
                    companion_bytes[destination_name]=(companion,companion.read_bytes())
                    document['artifacts'].append({'label':label,'url':f'./{destination_name}'})
        payload['research'][key]=document
        bindings[key]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
    payload['meta']['publication_status']='unpublished'
    payload['meta'].pop('publication_review',None)
    payload['meta'].pop('generation_manifest_sha256',None)
    payload['model']['lifecycle']['publication']={'status':'unpublished'}
    payload['model']['lifecycle']['deployment']={'status':'candidate'}
    for key in ('weekly','forecast','selection','label'):
        if payload[key]!=original[key]: raise ValueError(f'preview changed frozen {key}')
    validate_v5_payload(payload)
    raw=encoded(payload)
    core,research=build_dashboard_split(payload,payload_raw=raw)
    history,_=build_history_chunks(payload,payload_raw=raw)
    files={'data/regime-results.json':raw,'data/regime-core.json':core,
           'data/regime-research.json':research,**history}
    for key,raw_research in research_bytes.items(): files[f'data/{key}.json']=raw_research
    for name,(_,raw_companion) in companion_bytes.items(): files[name]=raw_companion
    comparison=json.loads((source.parent/'v5-vs-v4-comparison.json').read_text())
    comparison['inputs']['v5']['regime_results']['sha256']=hashlib.sha256(raw).hexdigest()
    files['data/v5-vs-v4-comparison.json']=encoded(comparison)
    files['data/selection-family-audit.json']=(source.parent/'selection-family-audit.json').read_bytes()
    for name in STATIC_ALLOWLIST: files[name]=(ROOT/'web'/name).read_bytes()
    files['index.html']=rewrite_index_asset_versions(files['index.html'],styles_raw=files['styles.css'],
        app_raw=files['app.js'],operating_contract_raw=files['operating-contract.generated.js'],
        extra_assets={name:files[name] for name in ('insights.js','insights.css')})
    inventory={name:{'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)} for name,data in files.items()}
    identity=hashlib.sha256(encoded(inventory)).hexdigest()
    generations=ROOT/'build'/'forecast-audit-improvements'/'preview-generations'
    generations.mkdir(parents=True,exist_ok=True)
    destination=generations/identity[:24]
    if not destination.exists():
        staging=Path(tempfile.mkdtemp(prefix='.staging-',dir=generations))
        try:
            for name,data in files.items():
                target=staging/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
                if hashlib.sha256(target.read_bytes()).hexdigest()!=inventory[name]['sha256']:
                    raise ValueError('preview file checksum differs')
            (staging/'preview-inventory.json').write_bytes(encoded(inventory))
            staging.rename(destination)
        finally:
            if staging.exists():shutil.rmtree(staging)
    elif any((destination/name).read_bytes()!=data for name,data in files.items()):
        raise ValueError('existing preview generation differs')
    if source.read_bytes()!=source_raw:raise ValueError('source changed during preview build')
    for key,path in research_files.items():
        if path.read_bytes()!=research_bytes[key]:
            raise ValueError(f'{key} changed during preview build')
    for path,raw_companion in companion_bytes.values():
        if path.read_bytes()!=raw_companion:
            raise ValueError('information companion changed during preview build')
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists() and not output.is_symlink():
        raise ValueError('preview output already exists as a real directory; choose another local path')
    pointer=output.parent/('.preview-'+uuid.uuid4().hex)
    pointer.symlink_to(os.path.relpath(destination,output.parent),target_is_directory=True)
    os.replace(pointer,output)
    result={'ok':True,'output':str(output),'resolved_output':str(destination),
        'source_sha256':hashlib.sha256(source_raw).hexdigest(),'source_unchanged':True,
        'official_weekly_and_selection_unchanged':True,'research_blocks':bindings,
        'preview_inventory_sha256':identity,'files':len(files),'external_publication':False}
    (output.parent/'preview-build.json').write_bytes(encoded(result))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'publication/live/regime-results.json')
    parser.add_argument('--forecast-research',type=Path)
    parser.add_argument('--calibration',type=Path)
    parser.add_argument('--information',type=Path)
    parser.add_argument('--operational',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'build/forecast-audit-improvements/preview')
    args=parser.parse_args()
    inputs={'forecast_research':args.forecast_research,'calibration_audit':args.calibration,
        'forecast_information':args.information,'operational_diagnostics':args.operational}
    print(json.dumps(build_preview(args.source,{k:v for k,v in inputs.items() if v is not None},args.output),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
