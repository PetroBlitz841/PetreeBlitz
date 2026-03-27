import { useState, useEffect } from "react";
import { isTiffUrl, convertTiffUrlToPng } from "../utils/tiff";

/**
 * Resolves an image src. If the URL points to a TIFF file, fetches and
 * converts it to a PNG data URL since browsers can't render TIFF natively.
 * Returns `null` while the conversion is in progress.
 */
export function useTiffAwareSrc(src: string): string | null {
  // Stores the last successfully converted TIFF: { src -> dataUrl }
  const [tiffConverted, setTiffConverted] = useState<{
    src: string;
    dataUrl: string;
  } | null>(null);

  useEffect(() => {
    // Non-TIFF sources need no async work — no setState here.
    if (!isTiffUrl(src)) return;

    let cancelled = false;

    convertTiffUrlToPng(src)
      .then((dataUrl) => {
        if (!cancelled) setTiffConverted({ src, dataUrl });
      })
      .catch(() => {
        // Fall back to the original URL on error
        if (!cancelled) setTiffConverted({ src, dataUrl: src });
      });

    return () => {
      cancelled = true;
    };
  }, [src]);

  // Non-TIFF: return src directly (no state involved).
  if (!isTiffUrl(src)) return src;
  // TIFF that has already been converted for this exact src:
  if (tiffConverted?.src === src) return tiffConverted.dataUrl;
  // Still converting — caller should show a skeleton.
  return null;
}
