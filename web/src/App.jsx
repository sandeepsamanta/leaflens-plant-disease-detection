import { useCallback, useEffect, useRef, useState } from 'react';
import { getModelCard, predict } from './api.js';
import History from './components/History.jsx';
import Reliability from './components/Reliability.jsx';
import Result from './components/Result.jsx';
import Uploader from './components/Uploader.jsx';
import { downscale, thumbnail } from './image.js';
import { checkFile, pushHistory } from './lib.js';

const HISTORY_KEY = 'leaflens.history';
const readHistory = () => {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
  } catch {
    return [];
  }
};
const saveHistory = (h) => {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(h));
  } catch {
    /* storage may be blocked */
  }
};

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [card, setCard] = useState(null);
  const [cardError, setCardError] = useState(false);
  const [samples, setSamples] = useState([]);
  const [history, setHistory] = useState(readHistory);
  const abort = useRef(null);
  const resultRef = useRef(null);

  useEffect(() => {
    getModelCard().then(setCard).catch(() => setCardError(true));
    fetch('/samples/manifest.json').then((r) => (r.ok ? r.json() : [])).then(setSamples).catch(() => {});
  }, []);

  useEffect(() => () => previewUrl && URL.revokeObjectURL(previewUrl), [previewUrl]);

  const choose = useCallback((f) => {
    const problem = checkFile(f);
    setResult(null);
    if (problem) {
      setError(problem);
      return;
    }
    setError('');
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
  }, []);

  const clear = () => {
    abort.current?.abort();
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError('');
    setBusy(false);
  };

  const analyse = async (f = file) => {
    if (!f) return;
    abort.current?.abort();
    const ctl = new AbortController();
    abort.current = ctl;
    setBusy(true);
    setError('');
    try {
      const res = await predict(await downscale(f), { signal: ctl.signal });
      setResult(res);
      const entry = {
        id: `${Date.now()}`,
        thumb: await thumbnail(f),
        plant: res.prediction.plant,
        disease: res.prediction.disease,
        confidence: res.prediction.confidence,
        decision: res.decision,
      };
      setHistory((h) => {
        const next = pushHistory(h, entry);
        saveHistory(next);
        return next;
      });
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50);
    } catch (e) {
      if (e.name !== 'AbortError') setError(e.message);
    } finally {
      if (abort.current === ctl) setBusy(false);
    }
  };

  const useSample = async (s) => {
    try {
      const blob = await (await fetch(`/samples/${s.file}`)).blob();
      const f = new File([blob], s.file, { type: blob.type || 'image/jpeg' });
      choose(f);
      analyse(f);
    } catch {
      setError('Could not load that sample.');
    }
  };

  return (
    <>
      <header className="top">
        <div className="wrap">
          <span className="logo" aria-hidden="true">🌿</span>
          <div>
            <h1>LeafLens</h1>
            <p>Plant disease detection with deep learning</p>
          </div>
        </div>
      </header>
      <main className="wrap">
        <Uploader previewUrl={previewUrl} busy={busy} onFile={choose} onAnalyse={() => analyse()} onClear={clear} samples={samples} onSample={useSample} />
        {error && <div className="card error" role="alert">{error}</div>}
        {busy && <div className="card loading" role="status"><span className="spinner" aria-hidden="true" /> Looking at the leaf...</div>}
        <div ref={resultRef}>{result && !busy && <Result result={result} previewUrl={previewUrl} />}</div>
        <Reliability card={card} error={cardError} />
        <History items={history} onClear={() => { setHistory([]); saveHistory([]); }} />
      </main>
      <footer className="wrap foot">
        <p>
          Trained on PlantVillage (CC BY-SA 3.0); field accuracy measured on PlantDoc (CC BY 4.0). Model: MobileNetV2 via transfer learning. Not a substitute for expert advice.
        </p>
      </footer>
    </>
  );
}
