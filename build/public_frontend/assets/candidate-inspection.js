(function () {
  'use strict';
  const core = window.MosaicCandidateInspection;
  const $ = selector => document.querySelector(selector);
  const create = (tag, className, content) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (content !== undefined) node.textContent = content;
    return node;
  };
  const link = (href, label, className) => {
    const node = create('a', className, label);
    node.href = href;
    return node;
  };
  const route = () => core.safePaperId(new URLSearchParams(location.search).get('paper'));
  const sourceUrl = (paper, page) => 'sources/' + encodeURIComponent(paper) + '.pdf' + (page ? '#page=' + page : '');
  const readerUrl = paper => 'index.html?paper=' + encodeURIComponent(paper) + '#papers';

  function setState(kind, message) {
    const box = $('#candidate-state');
    box.className = 'candidate-state ' + kind;
    box.hidden = false;
    box.textContent = message;
  }

  function addMeta(details, label, value) {
    if (!value) return;
    const row = create('div', 'metadata-row');
    row.append(create('dt', '', label), create('dd', 'mono', value));
    details.append(row);
  }

  function renderSegment(segment, paper, hasSourcePdf) {
    const figure = create('figure', 'source-segment');
    const caption = create('figcaption', 'segment-caption');
    const bits = [];
    if (segment.page) bits.push('Page ' + segment.page);
    caption.append(create('span', '', bits.join(' · ') || 'Captured source segment'));
    if (segment.page && hasSourcePdf) caption.append(link(sourceUrl(paper, segment.page), 'Open original PDF', 'source-link'));
    figure.append(caption, create('blockquote', '', segment.text || '[No segment text was recorded.]'));
    if (segment.unitTextSha256 || segment.unitId) {
      const digest = create('details', 'segment-digest');
      digest.append(create('summary', '', 'Captured text identity'));
      if (segment.unitId) digest.append(create('p', '', 'Captured unit ' + segment.unitId));
      if (segment.start !== null && segment.end !== null) digest.append(create('p', '', 'Characters ' + segment.start + '–' + segment.end));
      if (segment.unitTextSha256) digest.append(create('p', '', 'Digest of the full captured text unit.'), create('code', '', segment.unitTextSha256));
      figure.append(digest);
    }
    return figure;
  }

  function renderRecord(record, index, paper) {
    const article = create('article', 'candidate-record');
    const top = create('header', 'record-header');
    top.append(create('p', 'record-type', core.recordType(record)), create('h2', '', 'Candidate ' + (index + 1)));
    top.append(create('p', 'review-status', core.reviewLabel(record)));
    const attribution = record.candidate_attribution;
    const attributionLabels = {
      cited_study: 'Cited prior work · this excerpt is not an empirical result produced by this article.',
      current_article_empirical: 'Model attribution: this article’s empirical evidence · not independently reviewed.',
      current_article_theoretical: 'Model attribution: this article’s theory or interpretation · not independently reviewed.'
    };
    top.append(create('p', 'record-note', attributionLabels[attribution] ||
      (attribution ? 'Model attribution: ' + String(attribution).replaceAll('_', ' ') + ' · not independently reviewed.'
        : 'Attribution not recorded · inspect the original source.')));
    article.append(top);
    const note = create('p', 'record-note', 'This is preserved candidate evidence. It has not been independently reviewed for scientific accuracy, semantic support, coverage, or ownership.');
    article.append(note);
    const groups = core.evidenceGroups(record);
    if (!groups.length) {
      article.append(create('p', 'empty-evidence', 'No source segments are available for direct inspection in this record.'));
    } else {
      const groupBox = create('div', 'evidence-groups');
      groups.forEach((group, groupIndex) => {
        const section = create('section', 'evidence-group');
        section.append(create('h3', '', group.legacy ? 'Legacy selected excerpts' : 'Evidence group ' + (groupIndex + 1)));
        section.append(create('p', 'group-note', group.legacy
          ? 'This legacy record does not preserve the original grouped chain. The excerpts are shown in stored order.'
          : 'Segments are shown in their preserved order within this source-evidence group.'));
        const hasSourcePdf = record.source_pdf_available !== false && /^[0-9a-f]{64}$/.test(record.source_pdf_sha256 || '');
        group.segments.forEach(segment => section.append(renderSegment(segment, paper, hasSourcePdf)));
        groupBox.append(section);
      });
      article.append(groupBox);
    }
    const provenance = record.candidate_provenance || {};
    const metadata = create('details', 'provenance');
    metadata.append(create('summary', '', 'Technical provenance'));
    const list = create('dl', 'metadata-list');
    addMeta(list, 'Record identity', record.finding_id || provenance.record_id);
    addMeta(list, 'Original record kind', provenance.original_record_kind);
    addMeta(list, 'Source PDF digest', record.source_pdf_sha256 || provenance.source_pdf_sha256);
    addMeta(list, 'Captured-source version digest', record.source_units_sha256 || provenance.new_source_units_sha256);
    addMeta(list, 'Original raw record digest', provenance.raw_record_sha256);
    addMeta(list, 'Original record path', provenance.record_path);
    metadata.append(list);
    article.append(metadata);
    return article;
  }

  function render(payload, paper) {
    const rows = core.recordRows(payload);
    if (rows.some(row => row.paper_id !== paper || row.scientifically_accepted !== false)) throw Error('These records do not match the requested candidate paper.');
    const heading = $('#candidate-heading');
    const citation = $('#candidate-citation');
    const first = rows[0] || {};
    heading.textContent = first.title || 'Candidate evidence for ' + paper;
    citation.textContent = first.citation || 'Source metadata was not returned with these candidate records.';
    $('#candidate-reader').href = readerUrl(paper);
    $('#candidate-pdf').href = sourceUrl(paper);
    $('#candidate-pdf').hidden = first.source_pdf_available === false || !/^[0-9a-f]{64}$/.test(first.source_pdf_sha256 || '');
    const expectedCount = payload.matching_papers?.find(row => row.paper_id === paper)?.record_count;
    $('#candidate-record-count').textContent = rows.length + ' candidate record' + (rows.length === 1 ? '' : 's') + ' shown' + (Number.isInteger(expectedCount) && expectedCount > rows.length ? ' of ' + expectedCount + '. This view is limited; additional records are not shown.' : ' for inspection');
    const container = $('#candidate-records');
    container.replaceChildren();
    if (!rows.length) {
      container.append(create('section', 'candidate-empty', 'No candidate records were returned for this paper. Source passages may still be available in the reader.'));
      return;
    }
    rows.forEach((record, index) => container.append(renderRecord(record, index, paper)));
  }

  async function load() {
    const paper = route();
    $('#candidate-back').href = 'collection.html?status=proposals#papers';
    if (!paper) {
      setState('notice', 'Choose a paper with candidate records from the collection to inspect its preserved source evidence.');
      return;
    }
    try {
      const manifestResponse = await fetch('platform-manifest.json', { cache: 'no-store' });
      if (!manifestResponse.ok) throw Error('The platform manifest is unavailable.');
      const manifest = await manifestResponse.json();
      const response = await fetch('/api/search?paper=' + encodeURIComponent(paper) + '&page_size=1', { cache: 'no-store' });
      if (!response.ok) throw Error('Candidate records are unavailable.');
      const payload = await response.json();
      if (!core.sameSnapshot(manifest, payload)) throw Error('The search index and this preview do not match. Reload after the preview is rebuilt.');
      render(payload, paper);
      $('#candidate-content').hidden = false;
    } catch (error) {
      setState('error', error.message || 'Candidate evidence could not be loaded.');
    }
  }

  document.addEventListener('DOMContentLoaded', load);
})();
