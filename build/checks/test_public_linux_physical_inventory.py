"""Exercise real wheel/OS metadata layouts found in the pinned Linux image."""
from importlib.metadata import Distribution
from pathlib import Path

import pytest

import mosaic.public_linux_orphan_dependencies as repair


def distribution(root: Path, name: str, version: str, *, legacy: bool = False,
                 duplicate_path: bool = False) -> Distribution:
    metadata = root / (f"{name}-{version}.egg-info" if legacy else f"{name}-{version}.dist-info")
    metadata.mkdir(parents=True)
    (metadata / ('PKG-INFO' if legacy else 'METADATA')).write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding='utf-8')
    if legacy:
        (metadata / 'top_level.txt').write_text('package\n', encoding='utf-8')
    else:
        content = f"{metadata.name}/METADATA,,\n{metadata.name}/RECORD,,\n"
        if duplicate_path:
            content += f"{metadata.name}/METADATA,,\n"
        (metadata / 'RECORD').write_text(content, encoding='utf-8')
    return Distribution.at(metadata)


def test_wheel_and_shadowed_os_metadata_are_both_preserved(tmp_path, monkeypatch):
    local, system = tmp_path / 'local', tmp_path / 'system'
    monkeypatch.setattr(repair, '_OS_DISTRIBUTION_ROOTS', (system,))
    active = distribution(local, 'cryptography', '50.0.2')
    shadowed = distribution(system, 'cryptography', '41.0.7', legacy=True)
    manifest = repair.collect_installed_file_manifest([active, shadowed])
    rows = manifest['distributions']
    assert len(rows) == 2
    assert rows['cryptography']['distribution_version'] == '50.0.2'
    legacy = next(value for key, value in rows.items() if key.startswith('cryptography@'))
    assert legacy['record_sha256'] is None
    assert legacy['distribution_version'] == '41.0.7'
    assert len(legacy['metadata_file_hashes']) == 2
    assert manifest['shared_opentelemetry_files'] == {}
    (Path(shadowed._path) / 'top_level.txt').write_text('changed\n', encoding='utf-8')
    after = repair.collect_installed_file_manifest([active, shadowed])
    assert after['distributions']['cryptography'] == rows['cryptography']
    assert after['distributions'] != rows


def test_recordless_distribution_outside_observed_os_root_refuses(tmp_path):
    row = distribution(tmp_path, 'example', '1', legacy=True)
    with pytest.raises(repair.OrphanRepairRefusal, match='unrecognized_recordless'):
        repair.collect_installed_file_manifest([row])


def test_single_file_os_egg_info_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(repair, '_OS_DISTRIBUTION_ROOTS', (tmp_path,))
    path = tmp_path / 'cryptography.egg-info'
    path.write_text('Metadata-Version: 2.1\nName: cryptography\nVersion: 41.0.7\n', encoding='utf-8')
    result = repair.collect_installed_file_manifest([Distribution.at(path)])
    row = result['distributions']['cryptography']
    assert row['distribution_version'] == '41.0.7'
    assert row['record_sha256'] is None
    assert row['paths'] == ['cryptography.egg-info']
    assert set(row['metadata_file_hashes']) == {'cryptography.egg-info'}


def test_two_physical_telemetry_installations_still_refuse(tmp_path):
    first = distribution(tmp_path / 'one', repair.TARGET, '0.66b1')
    second = distribution(tmp_path / 'two', repair.TARGET, '0.66b1')
    with pytest.raises(repair.OrphanRepairRefusal, match='ambiguous_telemetry'):
        repair.collect_installed_file_manifest([first, second])


def test_repeated_wheel_path_still_refuses(tmp_path):
    row = distribution(tmp_path, 'example', '1', duplicate_path=True)
    with pytest.raises(repair.OrphanRepairRefusal, match='duplicate_distribution_path'):
        repair.collect_installed_file_manifest([row])


def test_repeated_physical_metadata_location_still_refuses(tmp_path):
    row = distribution(tmp_path, 'example', '1')
    with pytest.raises(repair.OrphanRepairRefusal, match='duplicate_metadata_location'):
        repair.collect_installed_file_manifest([row, row])
