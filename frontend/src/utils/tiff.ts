import * as UTIF from "utif2";

export function isTiffFile(file: File): boolean {
  return file.type === "image/tiff" || /\.tiff?$/i.test(file.name);
}

export function isTiffUrl(url: string): boolean {
  return /\.tiff?(\?.*)?$/i.test(url);
}

async function decodeToDataUrl(buffer: ArrayBuffer): Promise<string> {
  const ifds = UTIF.decode(buffer);
  if (ifds.length === 0) throw new Error("No images found in TIFF");
  const ifd = ifds[0];
  UTIF.decodeImage(buffer, ifd);
  const rgba = UTIF.toRGBA8(ifd);
  const canvas = document.createElement("canvas");
  canvas.width = ifd.width;
  canvas.height = ifd.height;
  const ctx = canvas.getContext("2d")!;
  const imgData = ctx.createImageData(ifd.width, ifd.height);
  imgData.data.set(rgba);
  ctx.putImageData(imgData, 0, 0);
  return canvas.toDataURL("image/png");
}

export async function convertTiffFileToPng(file: File): Promise<string> {
  const buffer = await file.arrayBuffer();
  return decodeToDataUrl(buffer);
}

export async function convertTiffUrlToPng(url: string): Promise<string> {
  const response = await fetch(url);
  if (!response.ok)
    throw new Error(`Failed to fetch image: ${response.status}`);
  const buffer = await response.arrayBuffer();
  return decodeToDataUrl(buffer);
}
