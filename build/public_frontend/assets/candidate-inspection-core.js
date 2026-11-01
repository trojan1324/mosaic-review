(function (root, factory) {
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.MosaicCandidateInspection = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const text = value => typeof value === 'string' ? value : '';
  const finite = value => Number.isInteger(value) && value >= 0;

  function safePaperId(value) {
    return /^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$/.test(text(value)) ? value : '';
  }

  function sameSnapshot(manifest, payload) {
    return Boolean(manifest && payload && manifest.snapshot_id && payload.snapshot_id &&
      manifest.snapshot_id === payload.snapshot_id);
  }

  function recordType(record) {
    const kind = text(record?.candidate_provenance?.original_record_kind);
    return kind === 'qualifications' ? 'Qualification' : kind === 'findings' ? 'Finding' : 'Candidate record';
  }

  function reviewLabel(record) {
    return !text(record?.assessment_status) || text(record?.assessment_status) === 'unchecked'
      ? 'Review has not been completed'
      : 'Review status: ' + text(record?.assessment_status);
  }

  function groupedEvidence(record) {
    const original = record?.candidate_provenance?.original_evidence;
    if (!Array.isArray(original) || original.length === 0) return [];
    return original.map((group, groupIndex) => ({
      id: text(group?.id) || text(group?.group_id) || 'group-' + (groupIndex + 1),
      order: groupIndex,
      segments: Array.isArray(group?.segments) ? group.segments.map((segment, segmentIndex) => ({
        order: segmentIndex,
        page: finite(segment?.page) ? segment.page : null,
        unitId: text(segment?.unit_id),
        start: finite(segment?.start) ? segment.start : null,
        end: finite(segment?.end) ? segment.end : null,
        text: text(segment?.text),
        unitTextSha256: text(segment?.unit_text_sha256)
      })) : []
    }));
  }

  function legacyEvidence(record) {
    const segments = Array.isArray(record?.selected_evidence) ? record.selected_evidence : [];
    const pages = record?.selected_evidence_pages;
    const onlyPage = Array.isArray(pages) && pages.length === 1 && finite(pages[0]) && pages[0] > 0 ? pages[0] : null;
    return segments.length ? [{ id: 'legacy-selected-evidence', order: 0, legacy: true,
      segments: segments.map((segment, index) => ({
        order: index,
        page: finite(segment?.page) && segment.page > 0 ? segment.page : onlyPage,
        unitId: text(segment?.unit_id), start: finite(segment?.start) ? segment.start : null,
        end: finite(segment?.end) ? segment.end : null, text: text(segment?.text),
        unitTextSha256: text(segment?.unit_text_sha256)
      })) }] : [];
  }

  function evidenceGroups(record) {
    const grouped = groupedEvidence(record);
    return grouped.length ? grouped : legacyEvidence(record);
  }

  function recordRows(payload) {
    return Array.isArray(payload?.records) ? payload.records.map(row => row?.record || row).filter(Boolean) : [];
  }

  return { safePaperId, sameSnapshot, recordType, reviewLabel, evidenceGroups, recordRows };
});
