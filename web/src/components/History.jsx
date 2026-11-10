import { pct } from '../lib.js';

export default function History({ items, onClear }) {
  if (!items.length) return null;
  return (
    <section className="card history" aria-labelledby="hist-h">
      <div className="row between">
        <h2 id="hist-h">Recent checks (this browser only)</h2>
        <button className="btn small" onClick={onClear}>Clear</button>
      </div>
      <ul>
        {items.map((h) => (
          <li key={h.id}>
            {h.thumb && <img src={h.thumb} alt="" width="44" height="44" />}
            <div>
              <strong>{h.plant}: {h.disease}</strong>
              <span className="muted small">
                {pct(h.confidence)} {h.decision === 'confident' ? '' : '(not sure)'}
              </span>
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}
