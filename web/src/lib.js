// Small pure helpers (unit-tested in lib.test.js).

export const pct = (x, digits = 0) => `${(x * 100).toFixed(digits)}%`;

/** How the result should be presented: 'uncertain' | 'healthy' | 'disease'. */
export function bandFor(result) {
  if (!result) return null;
  if (result.decision !== 'confident') return 'uncertain';
  return result.prediction.healthy ? 'healthy' : 'disease';
}

export const BAND_TEXT = {
  uncertain: { title: 'Not sure about this photo', tone: 'warn' },
  healthy: { title: 'Looks healthy', tone: 'good' },
  disease: { title: 'Possible disease detected', tone: 'bad' },
};

/** Validate a file before upload. Returns an error string or null. */
export function checkFile(file, maxMb = 12) {
  if (!file) return 'Choose a photo first.';
  if (!/^image\/(jpeg|png|webp)$/.test(file.type)) return 'Please use a JPEG, PNG or WebP photo.';
  if (file.size > maxMb * 1024 * 1024) return `That photo is larger than ${maxMb} MB.`;
  return null;
}

/** Keep at most `n` history entries, newest first, no duplicate ids. */
export function pushHistory(list, entry, n = 6) {
  return [entry, ...list.filter((e) => e.id !== entry.id)].slice(0, n);
}
