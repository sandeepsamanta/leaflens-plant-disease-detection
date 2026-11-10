import { useRef, useState } from 'react';

export default function Uploader({ previewUrl, busy, onFile, onAnalyse, onClear, samples, onSample }) {
  const input = useRef(null);
  const [over, setOver] = useState(false);

  const drop = (e) => {
    e.preventDefault();
    setOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) onFile(f);
  };

  return (
    <section className="card upload" aria-labelledby="upload-h">
      <h2 id="upload-h">1. Add a leaf photo</h2>
      {previewUrl ? (
        <div className="preview">
          <img src={previewUrl} alt="Your leaf photo" />
          <div className="row">
            <button className="btn primary" disabled={busy} onClick={onAnalyse}>
              {busy ? 'Analysing...' : 'Analyse leaf'}
            </button>
            <button className="btn" disabled={busy} onClick={onClear}>
              Choose another
            </button>
          </div>
        </div>
      ) : (
        <div
          className={`drop ${over ? 'over' : ''}`}
          onDragOver={(e) => {
            e.preventDefault();
            setOver(true);
          }}
          onDragLeave={() => setOver(false)}
          onDrop={drop}
          onClick={() => input.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && input.current?.click()}
          aria-label="Choose or drop a leaf photo"
        >
          <span className="drop-icon" aria-hidden="true">🍃</span>
          <p>
            <strong>Drop a photo here</strong> or tap to choose / take one
          </p>
          <p className="muted small">Best results: one leaf, filling most of the frame, in daylight.</p>
        </div>
      )}
      <input
        ref={input}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        capture="environment"
        hidden
        onChange={(e) => {
          const f = e.target.files?.[0];
          e.target.value = '';
          if (f) onFile(f);
        }}
      />
      {samples.length > 0 && (
        <div className="samples">
          <p className="muted small">No leaf handy? Try a sample:</p>
          <div className="sample-row">
            {samples.map((s) => (
              <button key={s.file} className="sample" disabled={busy} onClick={() => onSample(s)} title={`${s.title} (${s.kind})`}>
                <img src={`/samples/${s.file}`} alt={s.title} loading="lazy" />
                <span>{s.kind}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
