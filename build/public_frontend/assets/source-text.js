/* Reading presentation only. Evidence strings, offsets, hashes and exports stay original.
 * Call once per source paragraph/item/fragment; its surrounding element owns structure.
 */
function sourceText(value) {
  return String(value ?? '')
    // A discretionary soft hyphen marks a layout split, not a lexical hyphen.
    .replace(/\u00ad[^\S\r\n]*(?:\r\n?|\n)\s*/g, '')
    .replace(/\u00ad/g, '')
    // Retain real hyphens, including compounds split across a PDF line.
    .replace(/([\p{L}\p{M}])([-\u2010])[^\S\r\n]*(?:\r\n?|\n)\s*(?=[\p{L}\p{M}])/gu, '$1$2')
    .replace(/\s+/gu, ' ')
    .trim();
}
