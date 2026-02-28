import { Box, Typography, Stack, Alert, CircularProgress, Grid } from "@mui/material";
import { Album } from "../types";
import AlbumCard from "../components/AlbumCard";
import { useNavigate } from "react-router-dom";
import { useEffect, useState, useCallback } from "react";
import api from "../services/api";

export default function AlbumsPage() {
  const [albums, setAlbums] = useState<Album[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  const fetchAlbums = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await api.get<Album[]>("/albums");
      setAlbums(resp.data);
    } catch (err) {
      console.error(err);
      setError("Failed to load albums. Please try again.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAlbums();
  }, [fetchAlbums]);

  const handleSelect = (album: Album) => {
    navigate(`/albums/${album.album_id}`);
  };

  return (
    <Box sx={{ flex: 1, display: "flex", justifyContent: "center", alignItems: "flex-start", width: "100%", px: 2, py: 4 }}>
      <Box sx={{ width: "100%", maxWidth: 1200 }}>
        <Stack direction="column" spacing={1} sx={{ mb: 4 }}>
          <Typography variant="h4" color="primary" fontWeight="bold">
            Image Albums
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Browse your tree sample collections
          </Typography>
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

        {!loading && !error && albums.length === 0 && (
          <Alert severity="info" sx={{ mb: 3 }}>
            No albums yet. Albums are created automatically when you provide feedback on identified trees.
          </Alert>
        )}

        {!loading && (
          <Grid container spacing={3}>
            {albums.map((album) => (
              <Grid size={{ xs: 12, sm: 6, md: 4 }} key={album.album_id}>
                <AlbumCard album={album} onClick={() => handleSelect(album)} />
              </Grid>
            ))}
          </Grid>
        )}
      </Box>
    </Box>
  );
}
