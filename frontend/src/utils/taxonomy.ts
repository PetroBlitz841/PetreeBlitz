export interface TaxonomyEntry {
  genus: string;
  family: string;
  order: string;
}

/** Taxonomic classification keyed by genus name. Covers all 27 genera in the 44-species dataset. */
const GENUS_TAXONOMY: Record<string, { family: string; order: string }> = {
  // Fabales
  Apuleia: { family: "Fabaceae", order: "Fabales" },
  Cedrelinga: { family: "Fabaceae", order: "Fabales" },
  Copaifera: { family: "Fabaceae", order: "Fabales" },
  Diplotropis: { family: "Fabaceae", order: "Fabales" },
  Dipteryx: { family: "Fabaceae", order: "Fabales" },
  Enterolobium: { family: "Fabaceae", order: "Fabales" },
  Hymenaea: { family: "Fabaceae", order: "Fabales" },
  Hymenolobium: { family: "Fabaceae", order: "Fabales" },
  Luitzelburgia: { family: "Fabaceae", order: "Fabales" },
  Mimosa: { family: "Fabaceae", order: "Fabales" },
  Parkia: { family: "Fabaceae", order: "Fabales" },
  Piptadenia: { family: "Fabaceae", order: "Fabales" },
  Poeppigia: { family: "Fabaceae", order: "Fabales" },
  Poincianella: { family: "Fabaceae", order: "Fabales" },
  Sclerolobium: { family: "Fabaceae", order: "Fabales" },
  Vatairea: { family: "Fabaceae", order: "Fabales" },

  // Malpighiales
  Byrsonima: { family: "Malpighiaceae", order: "Malpighiales" },
  Calophyllum: { family: "Calophyllaceae", order: "Malpighiales" },
  Croton: { family: "Euphorbiaceae", order: "Malpighiales" },
  Hieronyma: { family: "Phyllanthaceae", order: "Malpighiales" },
  Jatropha: { family: "Euphorbiaceae", order: "Malpighiales" },
  Pera: { family: "Peraceae", order: "Malpighiales" },
  Sapium: { family: "Euphorbiaceae", order: "Malpighiales" },

  // Myrtales
  Combretum: { family: "Combretaceae", order: "Myrtales" },
  Erisma: { family: "Vochysiaceae", order: "Myrtales" },
  Qualea: { family: "Vochysiaceae", order: "Myrtales" },
  Vochysia: { family: "Vochysiaceae", order: "Myrtales" },

  // Sapindales
  Astronium: { family: "Anacardiaceae", order: "Sapindales" },
  Trattinnickia: { family: "Burseraceae", order: "Sapindales" },

  // Rosales
  Cecropia: { family: "Urticaceae", order: "Rosales" },

  // Gentianales
  Aspidosperma: { family: "Apocynaceae", order: "Gentianales" },

  // Malvales
  Cochlospermum: { family: "Bixaceae", order: "Malvales" },

  // Celastrales
  Goupia: { family: "Goupiaceae", order: "Celastrales" },

  // Lamiales
  Jacaranda: { family: "Bignoniaceae", order: "Lamiales" },
  Tabebuia: { family: "Bignoniaceae", order: "Lamiales" },

  // Laurales
  Licaria: { family: "Lauraceae", order: "Laurales" },
  Mezilaurus: { family: "Lauraceae", order: "Laurales" },
  Ocotea: { family: "Lauraceae", order: "Laurales" },

  // Apiales
  Schefflera: { family: "Araliaceae", order: "Apiales" },
};

/**
 * Derive the taxonomic classification for an album.
 * Genus is extracted by splitting album_id on `_` and taking the first token.
 * Unknown genera fall back to "Unknown" family and order.
 */
export function getTaxonomy(albumId: string): TaxonomyEntry {
  const genus = albumId.split("_")[0];
  const entry = GENUS_TAXONOMY[genus];
  if (entry) {
    return { genus, family: entry.family, order: entry.order };
  }
  return { genus, family: "Unknown", order: "Unknown" };
}

/** Build a 3-level taxonomy tree from a list of albums. */
export type TaxonomyTree = Map<
  string,
  Map<
    string,
    Map<string, { albumId: string; name: string; numImages: number }[]>
  >
>;

export function buildTaxonomyTree<
  T extends { album_id: string; name: string; num_images: number },
>(albums: T[]): TaxonomyTree {
  const tree: TaxonomyTree = new Map();

  for (const album of albums) {
    const { genus, family, order } = getTaxonomy(album.album_id);

    if (!tree.has(order)) tree.set(order, new Map());
    const orderMap = tree.get(order)!;

    if (!orderMap.has(family)) orderMap.set(family, new Map());
    const familyMap = orderMap.get(family)!;

    if (!familyMap.has(genus)) familyMap.set(genus, []);
    familyMap.get(genus)!.push({
      albumId: album.album_id,
      name: album.name,
      numImages: album.num_images,
    });
  }

  // Sort each level alphabetically; push "Unknown" to the end
  const sortKeys = (map: Map<string, unknown>) => {
    const sorted = [...map.entries()].sort(([a], [b]) => {
      if (a === "Unknown") return 1;
      if (b === "Unknown") return -1;
      return a.localeCompare(b);
    });
    map.clear();
    for (const [k, v] of sorted) map.set(k, v as never);
  };

  sortKeys(tree);
  for (const orderMap of tree.values()) {
    sortKeys(orderMap);
    for (const familyMap of orderMap.values()) {
      sortKeys(familyMap);
    }
  }

  return tree;
}
