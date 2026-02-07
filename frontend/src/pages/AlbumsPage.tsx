import { useState, useEffect } from "react";
import axios from "axios";
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Stack,
  Alert,
  CircularProgress,
} from "@mui/material";
import { FolderOpen, ArrowForward } from "@mui/icons-material";

interface Album {
  album_id: string;
  name: string;
  num_images: number;
}

interface AlbumsPageProps {
  onSelectAlbum?: (albumId: string, albumName?: string) => void;
}

export default function AlbumsPage({ onSelectAlbum }: AlbumsPageProps) {
  const [albums, setAlbums] = useState<Album[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchAlbums();
  }, []);

  const fetchAlbums = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await axios.get("/api/albums");
      setAlbums(response.data);
    } catch (err) {
      console.error("Failed to fetch albums:", err);
      setError("Failed to load albums. Please try again.");
    } finally {
      setLoading(false);
    }
  };

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
        <Stack direction="column" spacing={1} sx={{ mb: 4 }}>
          <Typography variant="h4" color="primary" fontWeight="bold">
            Image Albums
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Browse your tree sample collections
          </Typography>
        </Stack>

        {/* Loading State */}
        {loading && (
          <Box sx={{ display: "flex", justifyContent: "center", py: 4 }}>
            <CircularProgress />
          </Box>
        )}

        {/* Error Message */}
        {error && !loading && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}

        {/* Empty State */}
        {!loading && !error && albums.length === 0 && (
          <Alert severity="info" sx={{ mb: 3 }}>
            No albums yet. Albums are created automatically when you provide
            feedback on identified trees.
          </Alert>
        )}

        {/* Albums Grid */}
        {!loading && (
          <Grid container spacing={3}>
            {albums.map((album) => (
              <Grid size={{ xs: 12, sm: 6, md: 4 }} key={album.album_id}>
                <Card
                  sx={{
                    height: "100%",
                    display: "flex",
                    flexDirection: "column",
                    transition: "transform 0.2s, box-shadow 0.2s",
                    cursor: "pointer",
                    "&:hover": {
                      transform: "translateY(-4px)",
                      boxShadow: 3,
                    },
                  }}
                  onClick={() => onSelectAlbum?.(album.album_id, album.name)}
                >
                  {/* Placeholder */}
                  <Box
                    sx={{
                      height: 200,
                      backgroundColor: "grey.200",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                    }}
                  >
                    <FolderOpen sx={{ fontSize: 60, color: "grey.400" }} />
                  </Box>

                  {/* Card Content */}
                  <CardContent sx={{ flex: 1 }}>
                    <Stack
                      direction="row"
                      justifyContent="space-between"
                      alignItems="flex-start"
                      sx={{ mb: 2 }}
                    >
                      <Typography
                        variant="h6"
                        fontWeight="bold"
                        sx={{
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          whiteSpace: "nowrap",
                          flex: 1,
                        }}
                      >
                        {album.name}
                      </Typography>
                    </Stack>

                    <Stack
                      direction="row"
                      justifyContent="space-between"
                      alignItems="center"
                      sx={{
                        pt: 1,
                        borderTop: 1,
                        borderColor: "grey.200",
                      }}
                    >
                      <Typography variant="caption" color="text.secondary">
                        <strong>{album.num_images}</strong> image
                        {album.num_images !== 1 ? "s" : ""}
                      </Typography>
                      <ArrowForward
                        fontSize="small"
                        sx={{ color: "primary.main" }}
                      />
                    </Stack>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        )}
      </Box>
    </Box>
  );
}
