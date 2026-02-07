import { useState, useEffect } from "react";
import axios from "axios";
import {
  Box,
  Typography,
  Button,
  Card,
  CardContent,
  CardMedia,
  Grid,
  Stack,
  Alert,
  CircularProgress,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from "@mui/material";
import { ArrowBack, Info } from "@mui/icons-material";

interface Image {
  sample_id: string;
  image_url: string;
  predictions: Array<{
    label: string;
    confidence: number;
  }>;
  feedback?: {
    was_correct: boolean;
    correct_label?: string;
  };
  timestamp: string;
}

interface AlbumDetailsPageProps {
  albumId: string;
  albumName?: string;
  onBack: () => void;
}

export default function AlbumDetailsPage({
  albumId,
  albumName,
  onBack,
}: AlbumDetailsPageProps) {
  const [images, setImages] = useState<Image[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedImage, setSelectedImage] = useState<Image | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);

  useEffect(() => {
    fetchAlbumImages();
  }, [albumId]);

  const fetchAlbumImages = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await axios.get(`/api/albums/${albumId}/images`);
      setImages(response.data);
    } catch (err) {
      console.error("Failed to fetch album images:", err);
      setError("Failed to load images. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleViewDetails = (image: Image) => {
    setSelectedImage(image);
    setDetailsOpen(true);
  };

  const handleCloseDetails = () => {
    setDetailsOpen(false);
    setSelectedImage(null);
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence > 0.8) return "success";
    if (confidence > 0.5) return "warning";
    return "error";
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
        {/* Header with back button */}
        <Stack direction="row" spacing={2} alignItems="center" sx={{ mb: 4 }}>
          <Button startIcon={<ArrowBack />} onClick={onBack} variant="outlined">
            Back to Albums
          </Button>
          <Stack direction="column" spacing={0}>
            <Typography variant="h4" color="primary" fontWeight="bold">
              {albumName || albumId}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {images.length} {images.length === 1 ? "image" : "images"}
            </Typography>
          </Stack>
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
        {!loading && !error && images.length === 0 && (
          <Alert severity="info" sx={{ mb: 3 }}>
            No images in this album yet.
          </Alert>
        )}

        {/* Images Grid */}
        {!loading && (
          <Grid container spacing={3}>
            {images.map((image) => (
              <Grid size={{ xs: 12, sm: 6, md: 4 }} key={image.sample_id}>
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
                  onClick={() => handleViewDetails(image)}
                >
                  {/* Image */}
                  <CardMedia
                    component="img"
                    height="250"
                    image={`http://localhost:8000${image.image_url}`}
                    alt="Tree sample"
                    sx={{ objectFit: "cover" }}
                  />

                  {/* Card Content */}
                  <CardContent sx={{ flex: 1 }}>
                    {/* Top Prediction */}
                    {image.predictions.length > 0 && (
                      <Stack spacing={1} sx={{ mb: 2 }}>
                        <Typography variant="body2" color="text.secondary">
                          Top Prediction
                        </Typography>
                        <Typography variant="h6" fontWeight="bold">
                          {image.predictions[0].label}
                        </Typography>
                        <Chip
                          label={`${(image.predictions[0].confidence * 100).toFixed(1)}%`}
                          color={getConfidenceColor(
                            image.predictions[0].confidence,
                          )}
                          size="small"
                        />
                      </Stack>
                    )}

                    {/* Feedback Status */}
                    {image.feedback && (
                      <Box
                        sx={{ pt: 1, borderTop: 1, borderColor: "grey.200" }}
                      >
                        <Stack direction="row" alignItems="center" spacing={1}>
                          <Info fontSize="small" color="info" />
                          <Typography variant="caption" color="text.secondary">
                            {image.feedback.was_correct
                              ? "Correct prediction"
                              : `Corrected to: ${image.feedback.correct_label}`}
                          </Typography>
                        </Stack>
                      </Box>
                    )}

                    {/* Timestamp */}
                    <Typography
                      variant="caption"
                      color="text.secondary"
                      sx={{ mt: 1, display: "block" }}
                    >
                      {new Date(image.timestamp).toLocaleString()}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        )}

        {/* Image Details Dialog */}
        {selectedImage && (
          <Dialog
            open={detailsOpen}
            onClose={handleCloseDetails}
            maxWidth="md"
            fullWidth
          >
            <DialogTitle>Image Details</DialogTitle>
            <DialogContent sx={{ pt: 2 }}>
              <Stack spacing={3}>
                {/* Image */}
                <Box
                  component="img"
                  src={selectedImage.image_url}
                  alt="Tree sample"
                  sx={{ width: "100%", maxHeight: 400, objectFit: "contain" }}
                />

                {/* All Predictions */}
                <Stack spacing={1}>
                  <Typography variant="subtitle1" fontWeight="bold">
                    All Predictions
                  </Typography>
                  {selectedImage.predictions.map((pred, idx) => (
                    <Box
                      key={idx}
                      sx={{
                        p: 2,
                        backgroundColor: "grey.50",
                        borderRadius: 1,
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                      }}
                    >
                      <Typography>
                        {idx + 1}. {pred.label}
                      </Typography>
                      <Chip
                        label={`${(pred.confidence * 100).toFixed(1)}%`}
                        color={getConfidenceColor(pred.confidence)}
                      />
                    </Box>
                  ))}
                </Stack>

                {/* Feedback Information */}
                {selectedImage.feedback && (
                  <Stack spacing={1}>
                    <Typography variant="subtitle1" fontWeight="bold">
                      Feedback
                    </Typography>
                    <Box
                      sx={{ p: 2, backgroundColor: "blue.50", borderRadius: 1 }}
                    >
                      <Typography variant="body2">
                        {selectedImage.feedback.was_correct
                          ? "✓ Prediction was marked as correct"
                          : `✓ Correction provided: ${selectedImage.feedback.correct_label}`}
                      </Typography>
                    </Box>
                  </Stack>
                )}

                {/* Metadata */}
                <Stack spacing={1}>
                  <Typography variant="subtitle1" fontWeight="bold">
                    Metadata
                  </Typography>
                  <Box
                    sx={{ p: 2, backgroundColor: "grey.50", borderRadius: 1 }}
                  >
                    <Typography variant="caption" display="block">
                      <strong>Sample ID:</strong> {selectedImage.sample_id}
                    </Typography>
                    <Typography
                      variant="caption"
                      display="block"
                      sx={{ mt: 1 }}
                    >
                      <strong>Timestamp:</strong>{" "}
                      {new Date(selectedImage.timestamp).toLocaleString()}
                    </Typography>
                  </Box>
                </Stack>
              </Stack>
            </DialogContent>
            <DialogActions>
              <Button onClick={handleCloseDetails}>Close</Button>
            </DialogActions>
          </Dialog>
        )}
      </Box>
    </Box>
  );
}
