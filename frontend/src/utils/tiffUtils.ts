import UTIF from "utif";

/**
 * Returns true if the given file is a TIFF image based on its MIME type or file extension.
 */
export function isTiff(file: File): boolean {
  if (file.type === "image/tiff") return true;
  const lower = file.name.toLowerCase();
  return lower.endsWith(".tiff") || lower.endsWith(".tif");
}

/**
 * Decodes a TIFF File and returns a PNG data URL suitable for use in an <img> src.
 * Throws if decoding fails.
 */
export async function tiffFileToDataUrl(file: File): Promise<string> {
  const buffer = await file.arrayBuffer();
  const ifds = UTIF.decode(buffer);
  if (ifds.length === 0) throw new Error("No images found in TIFF file");
  UTIF.decodeImage(buffer, ifds[0]);
  const rgba = UTIF.toRGBA8(ifds[0]);

  const canvas = document.createElement("canvas");
  canvas.width = ifds[0].width;
  canvas.height = ifds[0].height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Unable to get 2D canvas context");

  const imageData = new ImageData(
    new Uint8ClampedArray(rgba.buffer as ArrayBuffer),
    ifds[0].width,
    ifds[0].height,
  );
  ctx.putImageData(imageData, 0, 0);
  return canvas.toDataURL("image/png");
}
