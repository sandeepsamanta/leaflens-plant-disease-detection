const BASE = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '');

async function parse(res) {
  let body = null;
  try {
    body = await res.json();
  } catch {
    /* not JSON */
  }
  if (!res.ok) {
    const detail = body && typeof body.detail === 'string' ? body.detail : null;
    throw new Error(detail || `Request failed (${res.status}). Please try again.`);
  }
  return body;
}

export async function predict(blob, { signal } = {}) {
  const form = new FormData();
  form.append('file', blob, blob.name || 'leaf.jpg');
  try {
    return await parse(await fetch(`${BASE}/api/predict?explain=true&top_k=3`, { method: 'POST', body: form, signal }));
  } catch (e) {
    if (e.name === 'AbortError') throw e;
    if (e instanceof TypeError) throw new Error('Could not reach the server. Check your connection and that the API is running.');
    throw e;
  }
}

export async function getModelCard() {
  return parse(await fetch(`${BASE}/api/model`));
}
