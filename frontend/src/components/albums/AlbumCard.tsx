import { Card, CardContent, Box, Stack, Typography, Chip } from "@mui/material";
import ParkIcon from "@mui/icons-material/Park";
import PhotoLibraryIcon from "@mui/icons-material/PhotoLibrary";
import ArrowForwardIcon from "@mui/icons-material/ArrowForward";
import { Album } from "../../types";
import { getGradient, splitSpeciesName } from "../../utils/albumStyle";

interface AlbumCardProps {
  album: Album;
  onClick?: () => void;
}

export default function AlbumCard({ album, onClick }: AlbumCardProps) {
  const [gradStart, gradEnd] = getGradient(album.album_id);
  const family = album.taxonomy?.family;
  const [genus, epithet] = splitSpeciesName(album.name);
  const initials = genus.slice(0, 2).toUpperCase();

  return (
    <Card
      sx={{
        height: "100%",
        display: "flex",
        flexDirection: "column",
        transition: "transform 0.18s ease, box-shadow 0.18s ease",
        cursor: onClick ? "pointer" : "default",
        borderRadius: 2,
        overflow: "hidden",
        "&:hover": onClick
          ? { transform: "translateY(-5px)", boxShadow: 6 }
          : undefined,
      }}
      onClick={onClick}
    >
      {/* ── Gradient banner ── */}
      <Box
        sx={{
          height: 140,
          background: `linear-gradient(135deg, ${gradStart} 0%, ${gradEnd} 100%)`,
          position: "relative",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          overflow: "hidden",
        }}
      >
        {/* Large semi-transparent monogram in background */}
        <Typography
          sx={{
            position: "absolute",
            fontSize: "7rem",
            fontWeight: 900,
            fontStyle: "italic",
            color: "rgba(255,255,255,0.12)",
            userSelect: "none",
            lineHeight: 1,
            letterSpacing: -4,
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
          }}
        >
          {initials}
        </Typography>

        {/* Centred leaf icon */}
        <ParkIcon
          sx={{ fontSize: 52, color: "rgba(255,255,255,0.85)", zIndex: 1 }}
        />

        {/* Image-count badge (top-right) */}
        <Stack
          direction="row"
          alignItems="center"
          spacing={0.5}
          sx={{
            position: "absolute",
            top: 10,
            right: 10,
            bgcolor: "rgba(0,0,0,0.35)",
            borderRadius: 5,
            px: 1,
            py: 0.3,
          }}
        >
          <PhotoLibraryIcon sx={{ fontSize: "0.8rem", color: "white" }} />
          <Typography
            sx={{ fontSize: "0.7rem", color: "white", fontWeight: 600 }}
          >
            {album.num_images}
          </Typography>
        </Stack>

        {/* Family chip (bottom-left) */}
        {family && family !== "Unknown" && (
          <Chip
            label={family}
            size="small"
            sx={{
              position: "absolute",
              bottom: 10,
              left: 10,
              bgcolor: "rgba(255,255,255,0.2)",
              color: "white",
              fontSize: "0.62rem",
              fontWeight: 600,
              backdropFilter: "blur(4px)",
              border: "1px solid rgba(255,255,255,0.35)",
            }}
          />
        )}
      </Box>

      {/* ── Content ── */}
      <CardContent sx={{ flex: 1, pb: "12px !important", pt: 1.5, px: 2 }}>
        {/* Species name */}
        <Typography variant="body1" sx={{ lineHeight: 1.35, mb: 0.5 }}>
          <Box component="span" fontStyle="italic" fontWeight={700}>
            {genus}
          </Box>
          {epithet && (
            <Box
              component="span"
              fontStyle="italic"
              fontWeight={400}
              color="text.secondary"
            >
              {" "}
              {epithet}
            </Box>
          )}
        </Typography>

        {/* Divider + footer row */}
        <Stack
          direction="row"
          justifyContent="space-between"
          alignItems="center"
          sx={{
            pt: 1,
            mt: "auto",
            borderTop: "1px solid",
            borderColor: "divider",
          }}
        >
          <Typography variant="caption" color="text.secondary">
            {album.num_images} {album.num_images === 1 ? "image" : "images"}
          </Typography>
          <ArrowForwardIcon fontSize="small" sx={{ color: "primary.main" }} />
        </Stack>
      </CardContent>
    </Card>
  );
}
