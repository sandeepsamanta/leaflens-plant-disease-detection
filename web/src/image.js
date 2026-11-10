// Downscale big phone photos before upload (saves bandwidth; the model only needs 224 px).
export async function downscale(file, maxSide = 1024, quality = 0.9) {
  try {
    const bmp = await createImageBitmap(file);
    const scale = Math.min(1, maxSide / Math.max(bmp.width, bmp.height));
    if (scale === 1 && file.size < 2 * 1024 * 1024) return file;
    const c = document.createElement('canvas');
    c.width = Math.round(bmp.width * scale);
    c.height = Math.round(bmp.height * scale);
    c.getContext('2d').drawImage(bmp, 0, 0, c.width, c.height);
    const blob = await new Promise((r) => c.toBlob(r, 'image/jpeg', quality));
    return blob ? new File([blob], 'leaf.jpg', { type: 'image/jpeg' }) : file;
  } catch {
    return file; // let the server decide
  }
}

export async function thumbnail(file, side = 72) {
  try {
    const bmp = await createImageBitmap(file);
    const s = Math.min(bmp.width, bmp.height);
    const c = document.createElement('canvas');
    c.width = c.height = side;
    c.getContext('2d').drawImage(bmp, (bmp.width - s) / 2, (bmp.height - s) / 2, s, s, 0, 0, side, side);
    return c.toDataURL('image/jpeg', 0.7);
  } catch {
    return null;
  }
}
