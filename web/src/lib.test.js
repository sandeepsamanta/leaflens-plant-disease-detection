import { describe, expect, it } from 'vitest';
import { bandFor, checkFile, pct, pushHistory } from './lib.js';

describe('bandFor', () => {
  it('is uncertain whenever the model is not confident', () => {
    expect(bandFor({ decision: 'uncertain', prediction: { healthy: true } })).toBe('uncertain');
  });
  it('separates healthy and diseased when confident', () => {
    expect(bandFor({ decision: 'confident', prediction: { healthy: true } })).toBe('healthy');
    expect(bandFor({ decision: 'confident', prediction: { healthy: false } })).toBe('disease');
  });
  it('handles no result', () => expect(bandFor(null)).toBeNull());
});

describe('checkFile', () => {
  it('rejects missing, wrong type and oversize files', () => {
    expect(checkFile(null)).toMatch(/Choose/);
    expect(checkFile({ type: 'application/pdf', size: 10 })).toMatch(/JPEG/);
    expect(checkFile({ type: 'image/jpeg', size: 20 * 1024 * 1024 })).toMatch(/larger/);
    expect(checkFile({ type: 'image/png', size: 1000 })).toBeNull();
  });
});

describe('pushHistory', () => {
  it('puts the newest first, drops duplicates and caps the length', () => {
    let h = [];
    for (let i = 0; i < 8; i++) h = pushHistory(h, { id: i });
    expect(h.map((e) => e.id)).toEqual([7, 6, 5, 4, 3, 2]);
    expect(pushHistory(h, { id: 5 }).map((e) => e.id)).toEqual([5, 7, 6, 4, 3, 2]);
  });
});

describe('pct', () => {
  it('formats fractions', () => expect(pct(0.9876, 1)).toBe('98.8%'));
});
