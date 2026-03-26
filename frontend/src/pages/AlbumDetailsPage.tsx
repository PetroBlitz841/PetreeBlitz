import React from "react";
import {
  Box,
  Typography,
  Stack,
  Alert,
  CircularProgress,
  Button,
  Chip,
  Divider,
  Tooltip,
  IconButton,
} from "@mui/material";
import Grid from "@mui/material/Grid";
import { ArrowBack } from "@mui/icons-material";
import AccountTreeIcon from "@mui/icons-material/AccountTree";
import FolderOpenIcon from "@mui/icons-material/FolderOpen";
import LocalFloristIcon from "@mui/icons-material/LocalFlorist";
import PhotoLibraryIcon from "@mui/icons-material/PhotoLibrary";
import OpenInNewIcon from "@mui/icons-material/OpenInNew";
import { useNavigate, useParams } from "react-router-dom";
import ImageCard from "../components/ImageCard";
import ImageDetailsDialog from "../components/ImageDetailsDialog";
import { Image } from "../types";
import { useEffect, useState, useCallback } from "react";
import api from "../services/api";
import { getTaxonomy } from "../utils/taxonomy";

// Shared palette with AlbumCard
const GRADIENTS = [
  ["#1b5e20", "#43a047"],
  ["#0d47a1", "#42a5f5"],
  ["#4a148c", "#9c27b0"],
  ["#bf360c", "#ff7043"],
  ["#006064", "#26c6da"],
  ["#37474f", "#78909c"],
  ["#880e4f", "#e91e63"],
  ["#33691e", "#8bc34a"],
];

function getGradient(albumId: string): [string, string] {
  const idx = albumId.charCodeAt(0) % GRADIENTS.length;
  return GRADIENTS[idx] as [string, string];
}

function albumIdToName(albumId: string) {
  return albumId.split("_").join(" ");
}

function splitName(name: string): [string, string] {
  const space = name.indexOf(" ");
  if (space === -1) return [name, ""];
  return [name.slice(0, space), name.slice(space + 1)];
}

function WikiButton({ name, label }: { name: string; label: string }) {
  const url = `https://en.wikipedia.org/wiki/${encodeURIComponent(name)}`;
  return (
    <Tooltip title={`"${name}" on Wikipedia`} placement="top">
      <IconButton
        size="small"
        component="a"
        href={url}
        target="_blank"
        rel="noopener noreferrer"
        sx={{ p: 0.4, color: "text.secondary", "&:hover": { color: "primary.main" } }}
        aria-label={label}
      >
        <OpenInNewIcon sx={{ fontSize: "0.88rem" }} />
      </IconButton>
    </Tooltip>
  );
}

export default function AlbumDetailsPage() {
  const { id: albumId } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [images, setImages] = useState<Image[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedImage, setSelectedImage] = React.useState<Image | null>(null);

  const fetchImages = useCallback(async () => {
    if (!albumId) return;
    setLoading(true);
    setError(null);
    try {
      const resp = await api.get<Image[]>(`/albums/${albumId}/images`);
      setImages(resp.data);
    } catch (err) {
      console.error(err);
      setError("Failed to load images. Please try again.");
    } finally {
      setLoading(false);
    }
  }, [albumId]);

  useEffect(() => {
    fetchImages();
  }, [fetchImages]);

  const handleBack = () => navigate("/albums");
  const handleView = (img: Image) => setSelectedImage(img);
  const handleClose = () => setSelectedImage(null);

  const safeId = albumId ?? "";
  const taxonomy = getTaxonomy(safeId);
  const [gradStart, gradEnd] = getGradient(safeId);
  const speciesName = albumIdToName(safeId);
  const [genus, epithet] = splitName(speciesName);
  const initials = genus.slice(0, 2).toUpperCase();

  return (
    <Box
      sx={{
        flex: 1,
        display: "flex",
        justifyContent: "center",
        alignItems: "flex-start",
        width: "100%",
        px: 2,
        py: 4,
      }}
    >
      <Box sx={{ width: "100%", maxWidth: 1200 }}>

        {/* ── Hero banner ── */}
        <Box
          sx={{
            borderRadius: 2,
            overflow: "hidden",
            mb: 3,
            background: `linear-gradient(135deg, ${gradStart} 0%, ${gradEnd} 100%)`,
            position: "relative",
            minHeight: 200,
            display: "flex",
            flexDirection: "column",
            justifyContent: "flex-end",
            p: 3,
          }}
        >
          {/* Monogram watermark */}
          <Typography
            sx={{
              position: "absolute",
              fontSize: "16rem",
              fontWeight: 900,
              fontStyle: "italic",
              color: "rgba(255,255,255,0.07)",
              userSelect: "none",
              lineHeight: 1,
              letterSpacing: -8,
              top: "50%",
              right: 32,
              transform: "translateY(-50%)",
              pointerEvents: "none",
            }}
          >
            {initials}
          </Typography>

          {/* Back button */}
          <Button
            startIcon={<ArrowBack />}
            onClick={handleBack}
            size="small"
            variant="outlined"
            sx={{
              position: "absolute",
              top: 16,
              left: 16,
              color: "rgba(255,255,255,0.9)",
              borderColor: "rgba(255,255,255,0.4)",
              "&:hover": { borderColor: "white", bgcolor: "rgba(255,255,255,0.12)" },
            }}
          >
            Albums
          </Button>

          {/* Image count badge */}
          {!loading && (
            <Stack
              direction="row"
              alignItems="center"
              spacing={0.5}
              sx={{
                position: "absolute",
                top: 16,
                right: 16,
                bgcolor: "rgba(0,0,0,0.35)",
                borderRadius: 5,
                px: 1.2,
                py: 0.4,
              }}
            >
              <PhotoLibraryIcon sx={{ fontSize: "0.85rem", color: "white" }} />
              <Typography sx={{ fontSize: "0.75rem", color: "white", fontWeight: 600 }}>
                {images.length} {images.length === 1 ? "image" : "images"}
              </Typography>
            </Stack>
          )}

          {/* Species name */}
          <Typography
            variant="h3"
            sx={{ color: "white", lineHeight: 1.2, textShadow: "0 2px 8px rgba(0,0,0,0.4)" }}
          >
            <Box component="span" fontStyle="italic" fontWeight={700}>
              {genus}
            </Box>
            {epithet && (
              <Box component="span" fontStyle="italic" fontWeight={300} sx={{ opacity: 0.85 }}>
                {" "}{epithet}
              </Box>
            )}
          </Typography>

          <Typography
            sx={{ color: "rgba(255,255,255,0.55)", fontSize: "0.78rem", mt: 0.5, fontFamily: "monospace" }}
          >
            {safeId}
          </Typography>
        </Box>

        {/* ── Taxonomy ribbon ── */}
        <Box
          sx={{
            mb: 3,
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
            <Chip
              label="Order"
              size="small"
              sx={{ fontSize: "0.58rem", height: 18, fontWeight: 700, letterSpacing: 0.6, bgcolor: "#bbdefb", color: "#0d47a1", textTransform: "uppercase" }}
            />
            <Typography variant="body2" fontWeight={600}>{taxonomy.order}</Typography>
            <WikiButton name={taxonomy.order} label={`${taxonomy.order} on Wikipedia`} />

            <Typography sx={{ color: "text.disabled", px: 0.25 }}>›</Typography>

            {/* Family */}
            <FolderOpenIcon sx={{ fontSize: "0.95rem", color: "secondary.main" }} />
            <Chip
              label="Family"
              size="small"
              sx={{ fontSize: "0.58rem", height: 18, fontWeight: 700, letterSpacing: 0.6, bgcolor: "#fff9c4", color: "#f57f17", textTransform: "uppercase" }}
            />
            <Typography variant="body2" fontWeight={600}>{taxonomy.family}</Typography>
            <WikiButton name={taxonomy.family} label={`${taxonomy.family} on Wikipedia`} />

            <Typography sx={{ color: "text.disabled", px: 0.25 }}>›</Typography>

            {/* Genus */}
            <LocalFloristIcon sx={{ fontSize: "0.95rem", color: "success.main" }} />
            <Chip
              label="Genus"
              size="small"
              sx={{ fontSize: "0.58rem", height: 18, fontWeight: 700, letterSpacing: 0.6, bgcolor: "#ffe0b2", color: "#bf360c", textTransform: "uppercase" }}
            />
            <Typography variant="body2" fontStyle="italic" fontWeight={600}>{taxonomy.genus}</Typography>
            <WikiButton name={taxonomy.genus} label={`${taxonomy.genus} on Wikipedia`} />

            <Typography sx={{ color: "text.disabled", px: 0.25 }}>›</Typography>

            {/* Species */}
            <Chip
              label="Species"
              size="small"
              sx={{ fontSize: "0.58rem", height: 18, fontWeight: 700, letterSpacing: 0.6, bgcolor: "#e8f5e9", color: "#2e7d32", textTransform: "uppercase" }}
            />
            <Typography variant="body2" fontStyle="italic">{speciesName}</Typography>
            <WikiButton name={speciesName} label={`${speciesName} on Wikipedia`} />

          </Stack>
        </Box>

        {/* ── Images section header ── */}
        <Stack direction="row" alignItems="center" spacing={1.5} sx={{ mb: 1 }}>
          <PhotoLibraryIcon sx={{ color: "text.secondary" }} />
          <Typography variant="h6" fontWeight={600}>
            Sample Images
          </Typography>
          {!loading && (
            <Chip label={images.length} size="small" color="primary" />
          )}
        </Stack>
        <Divider sx={{ mb: 3 }} />

        {loading && (
          <Box sx={{ display: "flex", justifyContent: "center", py: 4 }}>
            <CircularProgress />
          </Box>
        )}

        {error && !loading && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}

        {!loading && !error && images.length === 0 && (
          <Alert severity="info" sx={{ mb: 3 }}>
            No images in this album yet.
          </Alert>
        )}

        {!loading && (
          <Grid container spacing={3}>
            {images.map((image) => (
              <Grid size={{ xs: 12, sm: 6, md: 4 }} key={image.sample_id}>
                <ImageCard image={image} onClick={() => handleView(image)} />
              </Grid>
            ))}
          </Grid>
        )}

        <ImageDetailsDialog
          open={Boolean(selectedImage)}
          image={selectedImage}
          onClose={handleClose}
        />
      </Box>
    </Box>
  );
}

