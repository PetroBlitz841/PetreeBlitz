/**
 * Shared visual helpers for album-related components.
 * Keeps AlbumCard and AlbumDetailsPage in sync on gradient palette and name splitting.
 */

/** Deterministic gradient palette — indexed by genus initial char code */
export const GRADIENTS: [string, string][] = [
  ["#1b5e20", "#43a047"], // deep green
  ["#0d47a1", "#42a5f5"], // ocean blue
  ["#4a148c", "#9c27b0"], // violet
  ["#bf360c", "#ff7043"], // burnt orange
  ["#006064", "#26c6da"], // teal
  ["#37474f", "#78909c"], // slate
  ["#880e4f", "#e91e63"], // crimson
  ["#33691e", "#8bc34a"], // lime forest
];

/** Returns a [start, end] colour pair deterministic for an album_id. */
export function getGradient(albumId: string): [string, string] {
  const idx = albumId.charCodeAt(0) % GRADIENTS.length;
  return GRADIENTS[idx];
}

/**
 * Split a binomial species name ("Genus epithet") into [genus, epithet].
 * If there is no space the epithet is an empty string.
 */
export function splitSpeciesName(name: string): [string, string] {
  const space = name.indexOf(" ");
  if (space === -1) return [name, ""];
  return [name.slice(0, space), name.slice(space + 1)];
}
