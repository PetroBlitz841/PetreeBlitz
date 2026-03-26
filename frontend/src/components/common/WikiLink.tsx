import React from "react";
import { IconButton, Tooltip } from "@mui/material";
import OpenInNewIcon from "@mui/icons-material/OpenInNew";

interface WikiLinkProps {
  /** Taxon name — also used to build the Wikipedia URL. */
  name: string;
  /** Custom tooltip text. Defaults to `"${name}" on Wikipedia`. */
  tooltip?: string;
  /**
   * When true, clicks on the button also call stopPropagation.
   * Set this when the button is nested inside a clickable parent (e.g. Accordion).
   */
  stopPropagation?: boolean;
}

/**
 * Small icon-button that opens the Wikipedia article for a taxonomic name.
 * Used in TaxonomyTreeView (accordion rows) and TaxonomyBreadcrumb (detail page).
 */
export default function WikiLink({
  name,
  tooltip,
  stopPropagation = false,
}: WikiLinkProps) {
  const url = `https://en.wikipedia.org/wiki/${encodeURIComponent(name)}`;
  const title = tooltip ?? `"${name}" on Wikipedia`;

  const handleClick = stopPropagation
    ? (e: React.MouseEvent) => e.stopPropagation()
    : undefined;

  return (
    <Tooltip title={title} placement="top">
      <IconButton
        size="small"
        component="a"
        href={url}
        target="_blank"
        rel="noopener noreferrer"
        onClick={handleClick}
        sx={{
          p: 0.4,
          color: "text.secondary",
          "&:hover": { color: "primary.main" },
        }}
        aria-label={title}
      >
        <OpenInNewIcon sx={{ fontSize: "0.92rem" }} />
      </IconButton>
    </Tooltip>
  );
}
