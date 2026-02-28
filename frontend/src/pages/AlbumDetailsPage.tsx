import React from "react";
import { Box, Typography, Stack, Alert, CircularProgress, Grid, Button } from "@mui/material";
import { ArrowBack } from "@mui/icons-material";
import { useNavigate, useParams } from "react-router-dom";
import ImageCard from "../components/ImageCard";
import ImageDetailsDialog from "../components/ImageDetailsDialog";
import { Image } from "../types";
import { useEffect, useState, useCallback } from "react";
import api from "../services/api";

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

  return (
    <Box sx={{ flex: 1, display: "flex", justifyContent: "center", alignItems: "flex-start", width: "100%", px: 2, py: 4 }}>
      <Box sx={{ width: "100%", maxWidth: 1200 }}>
        <Stack direction="row" spacing={2} alignItems="center" sx={{ mb: 4 }}>
          <Button startIcon={<ArrowBack />} onClick={handleBack} variant="outlined">
            Back to Albums
          </Button>
          <Stack direction="column" spacing={0}>
            <Typography variant="h4" color="primary" fontWeight="bold">
              {albumId}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {images.length} {images.length === 1 ? "image" : "images"}
            </Typography>
          </Stack>
        </Stack>

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

        <ImageDetailsDialog open={Boolean(selectedImage)} image={selectedImage} onClose={handleClose} />
      </Box>
    </Box>
  );
}
