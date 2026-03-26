import { Chip } from "@mui/material";

export type TaxonomicRank = "Order" | "Family" | "Genus" | "Species";

const RANK_STYLES: Record<TaxonomicRank, { bgcolor: string; color: string }> = {
  Order: { bgcolor: "#bbdefb", color: "#0d47a1" },
  Family: { bgcolor: "#fff9c4", color: "#f57f17" },
  Genus: { bgcolor: "#ffe0b2", color: "#bf360c" },
  Species: { bgcolor: "#e8f5e9", color: "#2e7d32" },
};

interface TaxonomyRankChipProps {
  rank: TaxonomicRank;
}

/**
 * Small styled badge showing a taxonomic rank label.
 * Colours are consistent across TaxonomyTreeView and TaxonomyBreadcrumb.
 */
export default function TaxonomyRankChip({ rank }: TaxonomyRankChipProps) {
  const { bgcolor, color } = RANK_STYLES[rank];
  return (
    <Chip
      label={rank}
      size="small"
      variant="filled"
      sx={{
        fontSize: "0.58rem",
        height: 18,
        fontWeight: 700,
        letterSpacing: 0.6,
        bgcolor,
        color,
        textTransform: "uppercase",
      }}
    />
  );
}
