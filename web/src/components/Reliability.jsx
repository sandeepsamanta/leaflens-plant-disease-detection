import { pct } from '../lib.js';

export default function Reliability({ card, error }) {
  if (error) return null;
  if (!card) return <section className="card"><h2>How reliable is it?</h2><p className="muted">Loading model details...</p></section>;
  const lab = card.metrics.plantvillage_test;
  const field = card.metrics.plantdoc_field;
  return (
    <section className="card reliability" aria-labelledby="rel-h">
      <h2 id="rel-h">How reliable is it?</h2>
      <p className="muted">
        Two honest numbers. The model is far better on clean lab photos than on real field photos, so we show both.
      </p>
      <div className="stats">
        <div className="stat">
          <span className="stat-num">{pct(lab.accuracy, 1)}</span>
          <span className="stat-label">Lab photos (PlantVillage, leaf-grouped test, {lab.n.toLocaleString()} images)</span>
        </div>
        <div className="stat warn">
          <span className="stat-num">{pct(field.accuracy, 0)}</span>
          <span className="stat-label">
            Real field photos (PlantDoc, {field.n} leaves, 95% CI {pct(field.accuracy_ci95[0])} to {pct(field.accuracy_ci95[1])})
          </span>
        </div>
        <div className="stat">
          <span className="stat-num">{pct(field.plant_species_accuracy, 0)}</span>
          <span className="stat-label">Field photos: right crop identified</span>
        </div>
      </div>
      <details>
        <summary>Limitations</summary>
        <ul>
          {card.limitations.map((l) => (
            <li key={l}>{l}</li>
          ))}
        </ul>
      </details>
    </section>
  );
}
