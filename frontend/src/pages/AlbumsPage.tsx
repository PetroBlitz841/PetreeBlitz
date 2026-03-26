import {
  Box,
  Typography,
  Stack,
  Alert,
  CircularProgress,
  ToggleButton,
  ToggleButtonGroup,
} from "@mui/material";
import Grid from "@mui/material/Grid";
import GridViewIcon from "@mui/icons-material/GridView";
import AccountTreeIcon from "@mui/icons-material/AccountTree";
import { Album } from "../types";
import AlbumCard from "../components/albums/AlbumCard";
import TaxonomyTreeView from "../components/taxonomy/TaxonomyTreeView";
import { useNavigate } from "react-router-dom";
import { useEffect, useState, useCallback } from "react";
import api from "../services/api";
import { getTaxonomy } from "../utils/taxonomy";

export default function AlbumsPage() {
  const [albums, setAlbums] = useState<Album[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"taxonomy" | "grid">("taxonomy");
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

  /** Albums enriched with taxonomy so AlbumCard can show the family badge */
  const enriched: Album[] = albums.map((a) => ({
    ...a,
    taxonomy: getTaxonomy(a.album_id),
  }));

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
        {/* Header */}
        <Stack
          direction="row"
          justifyContent="space-between"
          alignItems="flex-start"
          sx={{ mb: 4 }}
          flexWrap="wrap"
          gap={2}
        >
          <Stack direction="column" spacing={0.5}>
            <Typography variant="h4" color="primary" fontWeight="bold">
              Image Albums
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Browse your tree sample collections
            </Typography>
          </Stack>

          {albums.length > 0 && (
            <ToggleButtonGroup
              exclusive
              value={viewMode}
              onChange={(_, v) => v && setViewMode(v)}
              size="small"
            >
              <ToggleButton value="taxonomy">
                <AccountTreeIcon fontSize="small" sx={{ mr: 0.75 }} />
                By Taxonomy
              </ToggleButton>
              <ToggleButton value="grid">
                <GridViewIcon fontSize="small" sx={{ mr: 0.75 }} />
                All Species
              </ToggleButton>
            </ToggleButtonGroup>
          )}
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
            No albums yet. Albums are created automatically when you provide
            feedback on identified trees.
          </Alert>
        )}

        {!loading && viewMode === "grid" && (
          <Grid container spacing={3}>
            {enriched.map((album) => (
              <Grid size={{ xs: 12, sm: 6, md: 4 }} key={album.album_id}>
                <AlbumCard album={album} onClick={() => handleSelect(album)} />
              </Grid>
            ))}
          </Grid>
        )}

        {!loading && viewMode === "taxonomy" && (
          <TaxonomyTreeView albums={enriched} onSelect={handleSelect} />
        )}
      </Box>
    </Box>
  );
}
