import { Box, Stack, Typography } from "@mui/material";
import AccountTreeIcon from "@mui/icons-material/AccountTree";
import FolderOpenIcon from "@mui/icons-material/FolderOpen";
import LocalFloristIcon from "@mui/icons-material/LocalFlorist";
import WikiLink from "../common/WikiLink";
import TaxonomyRankChip from "./TaxonomyRankChip";
import { TaxonomyEntry } from "../../types";

interface TaxonomyBreadcrumbProps {
  taxonomy: TaxonomyEntry;
  /** When provided, a fourth Species level is appended. */
  speciesName?: string;
}

function Separator() {
  return (
    <Typography
      component="span"
      sx={{ color: "text.disabled", px: 0.25, userSelect: "none" }}
    >
      ›
    </Typography>
  );
}

/**
 * Renders a horizontal taxonomy ribbon: Order → Family → Genus [→ Species].
 * Each rank shows a colour-coded badge, the taxon name, and a Wikipedia link.
 */
export default function TaxonomyBreadcrumb({
  taxonomy,
  speciesName,
}: TaxonomyBreadcrumbProps) {
  return (
    <Box
      sx={{
        px: 2.5,
        py: 1.5,
        bgcolor: "background.paper",
        borderRadius: 2,
        border: "1px solid",
        borderColor: "divider",
      }}
    >
      <Stack direction="row" alignItems="center" flexWrap="wrap" gap={0.75}>
        {/* Order */}
        <AccountTreeIcon sx={{ fontSize: "1rem", color: "primary.main" }} />
        <TaxonomyRankChip rank="Order" />
        <Typography variant="body2" fontWeight={600}>
          {taxonomy.order}
        </Typography>
        <WikiLink name={taxonomy.order} />

        <Separator />

        {/* Family */}
        <FolderOpenIcon sx={{ fontSize: "0.95rem", color: "secondary.main" }} />
        <TaxonomyRankChip rank="Family" />
        <Typography variant="body2" fontWeight={600}>
          {taxonomy.family}
        </Typography>
        <WikiLink name={taxonomy.family} />

        <Separator />

        {/* Genus */}
        <LocalFloristIcon sx={{ fontSize: "0.95rem", color: "success.main" }} />
        <TaxonomyRankChip rank="Genus" />
        <Typography variant="body2" fontStyle="italic" fontWeight={600}>
          {taxonomy.genus}
        </Typography>
        <WikiLink name={taxonomy.genus} />

        {/* Species */}
        {speciesName && (
          <>
            <Separator />
            <TaxonomyRankChip rank="Species" />
            <Typography variant="body2" fontStyle="italic">
              {speciesName}
            </Typography>
            <WikiLink name={speciesName} />
          </>
        )}
      </Stack>
    </Box>
  );
}
