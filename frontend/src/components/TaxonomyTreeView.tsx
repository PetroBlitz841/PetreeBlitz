import { useState } from "react";
import {
  Stack,
  Typography,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  IconButton,
  Tooltip,
} from "@mui/material";
import Grid from "@mui/material/Grid";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import AccountTreeIcon from "@mui/icons-material/AccountTree";
import FolderOpenIcon from "@mui/icons-material/FolderOpen";
import LocalFloristIcon from "@mui/icons-material/LocalFlorist";
import OpenInNewIcon from "@mui/icons-material/OpenInNew";
import { Album } from "../types";
import { buildTaxonomyTree } from "../utils/taxonomy";
import AlbumCard from "./AlbumCard";

/** Small Wikipedia link button — stops accordion toggle on click */
function WikiLink({ name, rank }: { name: string; rank: string }) {
  const url = `https://en.wikipedia.org/wiki/${encodeURIComponent(name)}`;
  return (
    <Tooltip title={`${rank} on Wikipedia`} placement="top">
      <IconButton
        size="small"
        component="a"
        href={url}
        target="_blank"
        rel="noopener noreferrer"
        onClick={(e: React.MouseEvent) => e.stopPropagation()}
        sx={{
          p: 0.4,
          color: "text.secondary",
          "&:hover": { color: "primary.main" },
        }}
      >
        <OpenInNewIcon sx={{ fontSize: "0.95rem" }} />
      </IconButton>
    </Tooltip>
  );
}

interface TaxonomyTreeViewProps {
  albums: Album[];
  onSelect: (album: Album) => void;
}

/** Counts total albums one level below a Map */
function countAlbums(
  map: Map<string, unknown[]> | Map<string, Map<string, unknown[]>>,
): number {
  let total = 0;
  for (const v of map.values()) {
    if (Array.isArray(v)) {
      total += v.length;
    } else {
      total += countAlbums(v as Map<string, unknown[]>);
    }
  }
  return total;
}

/** Small rank badge chip */
function RankChip({
  label,
  bgcolor,
  color,
}: {
  label: string;
  bgcolor: string;
  color: string;
}) {
  return (
    <Chip
      label={label}
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

export default function TaxonomyTreeView({
  albums,
  onSelect,
}: TaxonomyTreeViewProps) {
  const tree = buildTaxonomyTree(albums);

  // Track expanded state independently per level using "<order>", "<order>/<family>", "<order>/<family>/<genus>" keys
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const toggle = (key: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      // eslint-disable-next-line @typescript-eslint/no-unused-expressions
      next.has(key) ? next.delete(key) : next.add(key);
      return next;
    });

  if (albums.length === 0) return null;

  return (
    <Stack spacing={1}>
      {[...tree.entries()].map(([order, familyMap]) => {
        const orderKey = order;
        const orderTotal = countAlbums(
          familyMap as Map<string, Map<string, unknown[]>>,
        );

        return (
          <Accordion
            key={orderKey}
            expanded={expanded.has(orderKey)}
            onChange={() => toggle(orderKey)}
            disableGutters
            elevation={1}
            sx={{ "&:before": { display: "none" }, borderRadius: 1 }}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Stack
                direction="row"
                alignItems="center"
                spacing={1.5}
                width="100%"
              >
                <AccountTreeIcon
                  sx={{ fontSize: "1.1rem", color: "primary.main" }}
                />
                <Typography variant="subtitle1" fontWeight={700} flex={1}>
                  {order}
                </Typography>
                <RankChip label="Order" bgcolor="#bbdefb" color="#0d47a1" />
                <WikiLink name={order} rank="Order" />
                <Chip
                  label={`${familyMap.size} ${familyMap.size === 1 ? "family" : "families"}`}
                  size="small"
                  variant="outlined"
                  sx={{ fontSize: "0.7rem" }}
                />
                <Chip
                  label={`${orderTotal} ${orderTotal === 1 ? "species" : "species"}`}
                  size="small"
                  color="primary"
                  sx={{ fontSize: "0.7rem" }}
                />
              </Stack>
            </AccordionSummary>

            <AccordionDetails sx={{ pt: 0, pb: 1, px: 2 }}>
              <Stack spacing={1}>
                {[...familyMap.entries()].map(([family, genusMap]) => {
                  const familyKey = `${orderKey}/${family}`;
                  const familyTotal = countAlbums(
                    genusMap as Map<string, unknown[]>,
                  );

                  return (
                    <Accordion
                      key={familyKey}
                      expanded={expanded.has(familyKey)}
                      onChange={() => toggle(familyKey)}
                      disableGutters
                      elevation={0}
                      sx={{
                        border: "1px solid",
                        borderColor: "divider",
                        borderRadius: 1,
                        "&:before": { display: "none" },
                      }}
                    >
                      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                        <Stack
                          direction="row"
                          alignItems="center"
                          spacing={1.5}
                          width="100%"
                        >
                          <FolderOpenIcon
                            sx={{ fontSize: "1rem", color: "secondary.main" }}
                          />
                          <Typography variant="body1" fontWeight={600} flex={1}>
                            {family}
                          </Typography>
                          <RankChip
                            label="Family"
                            bgcolor="#fff9c4"
                            color="#f57f17"
                          />
                          <WikiLink name={family} rank="Family" />
                          <Chip
                            label={`${genusMap.size} ${genusMap.size === 1 ? "genus" : "genera"}`}
                            size="small"
                            variant="outlined"
                            sx={{ fontSize: "0.68rem" }}
                          />
                          <Chip
                            label={`${familyTotal} species`}
                            size="small"
                            color="secondary"
                            sx={{ fontSize: "0.68rem" }}
                          />
                        </Stack>
                      </AccordionSummary>

                      <AccordionDetails sx={{ pt: 0.5, pb: 1.5, px: 2 }}>
                        <Stack spacing={1.5}>
                          {[...genusMap.entries()].map(
                            ([genus, speciesList]) => {
                              const genusKey = `${familyKey}/${genus}`;

                              return (
                                <Accordion
                                  key={genusKey}
                                  expanded={expanded.has(genusKey)}
                                  onChange={() => toggle(genusKey)}
                                  disableGutters
                                  elevation={0}
                                  sx={{
                                    backgroundColor: "grey.50",
                                    border: "1px solid",
                                    borderColor: "divider",
                                    borderRadius: 1,
                                    "&:before": { display: "none" },
                                  }}
                                >
                                  <AccordionSummary
                                    expandIcon={<ExpandMoreIcon />}
                                  >
                                    <Stack
                                      direction="row"
                                      alignItems="center"
                                      spacing={1.5}
                                      width="100%"
                                    >
                                      <LocalFloristIcon
                                        sx={{
                                          fontSize: "0.95rem",
                                          color: "success.main",
                                        }}
                                      />
                                      <Typography
                                        variant="body2"
                                        fontStyle="italic"
                                        fontWeight={600}
                                        flex={1}
                                      >
                                        {genus}
                                      </Typography>
                                      <RankChip
                                        label="Genus"
                                        bgcolor="#ffe0b2"
                                        color="#bf360c"
                                      />
                                      <WikiLink name={genus} rank="Genus" />
                                      <Chip
                                        label={`${speciesList.length} ${speciesList.length === 1 ? "species" : "species"}`}
                                        size="small"
                                        variant="outlined"
                                        sx={{ fontSize: "0.65rem" }}
                                      />
                                    </Stack>
                                  </AccordionSummary>

                                  <AccordionDetails
                                    sx={{ pt: 1, pb: 1.5, px: 1.5 }}
                                  >
                                    <Grid container spacing={2}>
                                      {speciesList.map((sp) => {
                                        const album: Album = {
                                          album_id: sp.albumId,
                                          name: sp.name,
                                          num_images: sp.numImages,
                                          taxonomy: {
                                            genus,
                                            family,
                                            order,
                                          },
                                        };
                                        return (
                                          <Grid
                                            size={{ xs: 12, sm: 6, md: 4 }}
                                            key={sp.albumId}
                                          >
                                            <AlbumCard
                                              album={album}
                                              onClick={() => onSelect(album)}
                                            />
                                          </Grid>
                                        );
                                      })}
                                    </Grid>
                                  </AccordionDetails>
                                </Accordion>
                              );
                            },
                          )}
                        </Stack>
                      </AccordionDetails>
                    </Accordion>
                  );
                })}
              </Stack>
            </AccordionDetails>
          </Accordion>
        );
      })}
    </Stack>
  );
}
