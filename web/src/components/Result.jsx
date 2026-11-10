import { useState } from 'react';
import { BAND_TEXT, bandFor, pct } from '../lib.js';

export default function Result({ result, previewUrl }) {
  const [showHeat, setShowHeat] = useState(true);
  const band = bandFor(result);
  const { prediction, top, advice } = result;
  const meta = BAND_TEXT[band];

  return (
    <section className={`card result ${meta.tone}`} aria-live="polite" aria-labelledby="result-h">
      <h2 id="result-h">2. Result</h2>
      <div className={`banner ${meta.tone}`}>
        <strong>{meta.title}</strong>
      </div>
      {result.message && <p className="notice">{result.message}</p>}

      <div className="headline">
        <div>
          <p className="muted small">{band === 'uncertain' ? 'Closest match' : 'Prediction'}</p>
          <h3>
            {prediction.plant}: {prediction.disease}
          </h3>
        </div>
        <div className="conf" aria-label={`Confidence ${pct(prediction.confidence)}`}>
          <span className="conf-num">{pct(prediction.confidence)}</span>
          <span className="muted small">confidence</span>
        </div>
      </div>

      <ul className="bars" aria-label="Top predictions">
        {top.map((t, i) => (
          <li key={t.label}>
            <span className="bar-label">
              {t.plant}: {t.disease}
            </span>
            <span className="bar-track">
              <span className={`bar-fill ${i === 0 ? 'first' : ''}`} style={{ width: pct(t.confidence) }} />
            </span>
            <span className="bar-num">{pct(t.confidence)}</span>
          </li>
        ))}
      </ul>

      {result.heatmap && (
        <figure className="heat">
          <div className="heat-img">
            <img src={showHeat ? result.heatmap : previewUrl} alt={showHeat ? 'Heatmap of the areas the model focused on' : 'Your leaf photo'} />
          </div>
          <figcaption>
            <label className="switch">
              <input type="checkbox" checked={showHeat} onChange={(e) => setShowHeat(e.target.checked)} />
              Show what the model looked at
            </label>
            <span className="muted small">Warm colours mark the regions that pushed the model towards this answer. If they sit on the background, don't trust the result.</span>
          </figcaption>
        </figure>
      )}

      <div className="advice">
        <h3>About this</h3>
        <p>{advice.about}</p>
        <h4>What it looks like</h4>
        <p>{advice.symptoms}</p>
        <h4>What you can do</h4>
        <ul>
          {advice.actions.map((a) => (
            <li key={a}>{a}</li>
          ))}
        </ul>
      </div>
      <p className="disclaimer">
        LeafLens is a decision aid, not a diagnosis. Confirm with a local agricultural extension service before treating, especially for chemicals.
      </p>
    </section>
  );
}
