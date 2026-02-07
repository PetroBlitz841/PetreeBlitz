import React, { useState, useRef } from "react";
import axios from "axios";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import {
  Box,
  Typography,
  Button,
  Card,
  CardContent,
  CardMedia,
  CircularProgress,
  Alert,
  Grid,
  Chip,
  Stack,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Tabs,
  Tab,
  Paper,
} from "@mui/material";
import {
  CloudUpload,
  CheckCircle,
  Cancel,
  Collections,
} from "@mui/icons-material";
import AlbumsPage from "./pages/AlbumsPage";
import AlbumDetailsPage from "./pages/AlbumDetailsPage";

// Create a custom theme
const theme = createTheme({
  palette: {
    primary: {
      main: "#4caf50",
    },
    secondary: {
      main: "#8bc34a",
    },
  },
  typography: {
    fontFamily: "Heebo",
  },
});

interface Prediction {
  label: string;
  confidence: number;
}

interface Feedback {
  sample_id: string;
  was_correct: boolean;
  correct_label?: string;
}

type Page = "identify" | "albums" | "album-details";

function App() {
  // Page routing
  const [currentPage, setCurrentPage] = useState<Page>("identify");
  const [selectedAlbumId, setSelectedAlbumId] = useState<string | null>(null);
  const [selectedAlbumName, setSelectedAlbumName] = useState<string | null>(
    null,
  );

  // Identification page state
  const [file, setFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [results, setResults] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [sampleId, setSampleId] = useState<string | null>(null);
  const [feedbackLoading, setFeedbackLoading] = useState(false);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [correctLabel, setCorrectLabel] = useState("");
  const [pendingFeedback, setPendingFeedback] = useState<{
    label: string;
    correct: boolean;
  } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Keep track of blob URL to revoke it when replaced/cleared
  const previewUrlRef = useRef<string | null>(null);

  // Utility to safely revoke the previous preview URL
  const revokePreview = () => {
    if (previewUrlRef.current) {
      try {
        URL.revokeObjectURL(previewUrlRef.current);
      } catch {
        // ignore revoke errors
      }
      previewUrlRef.current = null;
    }
  };

  // Clear the current image preview
  const clearImagePreview = () => {
    revokePreview();
    setImagePreview(null);
  };

  // Handle a newly selected file
  const handleFileSelect = (selectedFile: File) => {
    revokePreview(); // revoke previous preview if present

    const previewUrl = URL.createObjectURL(selectedFile);
    previewUrlRef.current = previewUrl;

    setFile(selectedFile);
    setError(null);
    setResults([]);
    setImagePreview(previewUrl);

    // allow selecting the same file again
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  // Handle drag and drop
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  // Handle file input change
  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  // Identify tree species
  const identifyTree = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await axios.post("api/identify", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      console.log("API Response:", response.data);
      setSampleId(response.data.sample_id);
      setResults(response.data.predictions || []);
    } catch (err) {
      setError("Failed to identify tree species. Please try again.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Send feedback
  const sendFeedback = async (label: string, correct: boolean) => {
    if (!sampleId) {
      setError("Sample ID not available. Please try identifying again.");
      return;
    }

    if (!correct) {
      // Ask user for correct label if prediction was wrong
      setPendingFeedback({ label, correct });
      setDialogOpen(true);
      return;
    }

    // Send positive feedback immediately
    setFeedbackLoading(true);
    try {
      const payload = {
        sample_id: sampleId,
        was_correct: true,
      };

      const response = await axios.post("api/feedback", payload);
      console.log("Feedback response:", response.data);

      // Reset states
      setFile(null);
      clearImagePreview();
      setResults([]);
      setSampleId(null);

      // Show success message (basic - could be improved with a toast/snackbar)
      console.log("Feedback submitted successfully!");
    } catch (err) {
      console.error("Failed to send feedback:", err);
      setError("Failed to submit feedback. Please try again.");
    } finally {
      setFeedbackLoading(false);
    }
  };

  const submitFeedback = async (correct: boolean) => {
    if (!sampleId || !pendingFeedback) return;

    setFeedbackLoading(true);
    try {
      const payload: Feedback = {
        sample_id: sampleId,
        was_correct: correct,
      };

      // Only include correct_label if user provided label
      if (!correct && correctLabel.trim()) {
        payload.correct_label = correctLabel.trim();
      }

      const response = await axios.post("api/feedback", payload);
      console.log("Feedback response:", response.data);

      // Reset states
      setFile(null);
      clearImagePreview();
      setResults([]);
      setSampleId(null);
      setPendingFeedback(null);
      setCorrectLabel("");
      setDialogOpen(false);

      // Show success message
      console.log("Feedback submitted successfully!");
    } catch (err) {
      console.error("Failed to send feedback:", err);
      setError("Failed to submit feedback. Please try again.");
    } finally {
      setFeedbackLoading(false);
    }
  };

  const handleDialogClose = () => {
    setDialogOpen(false);
    setPendingFeedback(null);
    setCorrectLabel("");
  };

  const handleDialogSubmit = async () => {
    await submitFeedback(false);
  };

  const handleSelectAlbum = (albumId: string, albumName?: string) => {
    setSelectedAlbumId(albumId);
    setSelectedAlbumName(albumName || albumId);
    setCurrentPage("album-details");
  };

  const handleBackToAlbums = () => {
    setCurrentPage("albums");
    setSelectedAlbumId(null);
    setSelectedAlbumName(null);
  };

  const handlePageChange = (_: React.SyntheticEvent, newValue: Page) => {
    setCurrentPage(newValue);
    setSelectedAlbumId(null);
    setSelectedAlbumName(null);
  };

  return (
    <ThemeProvider theme={theme}>
      <Box
        sx={{
          minHeight: "100vh",
          display: "flex",
          flexDirection: "column",
          backgroundColor: "grey.50",
        }}
      >
        {/* Header */}
        <Box
          component="header"
          sx={{ backgroundColor: "white", boxShadow: 1, py: 2 }}
        >
          <Box sx={{ px: 2 }}>
            <Stack direction="row" spacing={2} alignItems="center">
              <img
                src="/pt-logo.svg"
                alt="PetreeBlitz Logo"
                style={{ height: "100px" }}
              />
              <Stack
                p={2}
                direction="column"
                spacing={0}
                alignItems="flex-start"
              >
                <Typography variant="h4" color="text.secondary">
                  Archaeobotany AI Tree Identification
                </Typography>
              </Stack>
            </Stack>
          </Box>

          {/* Navigation Tabs */}
          <Box sx={{ px: 2, mt: 2 }}>
            <Paper sx={{ backgroundColor: "transparent", boxShadow: "none" }}>
              <Tabs
                value={currentPage === "album-details" ? "albums" : currentPage}
                onChange={handlePageChange}
                variant="scrollable"
                scrollButtons="auto"
              >
                <Tab label="Identify Trees" value="identify" />
                <Tab
                  label="Albums"
                  value="albums"
                  icon={<Collections />}
                  iconPosition="start"
                />
              </Tabs>
            </Paper>
          </Box>
        </Box>

        {/* Main Content */}
        <Box
          sx={{
            flex: 1,
            display: "flex",
            justifyContent: "center",
            alignItems: "flex-start",
            width: "100%",
          }}
        >
          {/* Identify Page */}
          {currentPage === "identify" && (
            <Box
              sx={{
                flex: 1,
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
                width: "100%",
                px: 2,
              }}
            >
              <Box
                sx={{
                  width: "70%",
                  display: "flex",
                  justifyContent: "center",
                  alignItems: "center",
                  py: 4,
                }}
              >
                <Stack direction="column" spacing={3}>
                  <Stack direction="column" spacing={1}>
                    <Typography variant="h4" color="primary" textAlign="center">
                      Identify Tree Species from Images
                    </Typography>
                    <Typography
                      variant="h6"
                      textAlign="center"
                      color="text.secondary"
                    >
                      Upload a photo of a tree sample and our AI will identify
                      the species with confidence scores. Help improve our model
                      by providing feedback on predictions.
                    </Typography>
                  </Stack>

                  {/* Upload Area */}
                  <Card
                    sx={{
                      border: dragOver
                        ? "2px dashed #4caf50"
                        : "2px dashed #ccc",
                      backgroundColor: dragOver ? "green.50" : "white",
                      cursor: "pointer",
                      width: "100%",
                    }}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <CardContent sx={{ textAlign: "center", py: 4 }}>
                      <CloudUpload
                        sx={{ fontSize: 48, color: "primary.main" }}
                      />
                      <Typography variant="h6" sx={{ mt: 1 }}>
                        {file
                          ? file.name
                          : "Drag & drop an image here or click to select"}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Supported formats: JPG, PNG, GIF
                      </Typography>
                      <input
                        ref={fileInputRef}
                        type="file"
                        accept="image/*"
                        style={{ display: "none" }}
                        onChange={handleFileInputChange}
                      />
                    </CardContent>
                  </Card>

                  {/* Image Preview */}
                  {imagePreview && (
                    <Box display="flex" justifyContent="center">
                      <Card sx={{ width: "60%" }}>
                        <CardMedia
                          component="img"
                          height="300"
                          image={imagePreview}
                          alt="Uploaded tree sample"
                          sx={{ objectFit: "contain" }}
                        />
                      </Card>
                    </Box>
                  )}

                  {/* Identify Button */}
                  {file && (
                    <Box sx={{ textAlign: "center" }}>
                      <Button
                        variant="contained"
                        size="large"
                        onClick={identifyTree}
                        disabled={loading}
                        startIcon={
                          loading ? <CircularProgress size={20} /> : null
                        }
                      >
                        {loading ? "Identifying..." : "Identify Tree Species"}
                      </Button>
                    </Box>
                  )}

                  {/* Error Display */}
                  {error && <Alert severity="error">{error}</Alert>}

                  {/* Results Display */}
                  {results.length > 0 && (
                    <Stack
                      direction="column"
                      spacing={2}
                      sx={{ width: "100%" }}
                    >
                      <Typography variant="h5">
                        Identification Results
                      </Typography>
                      <Grid container spacing={2}>
                        {results.map((prediction, index) => (
                          <Grid size={{ xs: 12, sm: 4 }} key={index}>
                            <Card>
                              <CardContent>
                                <Typography variant="h6" fontWeight="bold">
                                  #{index + 1} {prediction.label}
                                </Typography>
                                <Chip
                                  label={`${(prediction.confidence * 100).toFixed(1)}%`}
                                  color={
                                    prediction.confidence > 0.8
                                      ? "success"
                                      : prediction.confidence > 0.5
                                        ? "warning"
                                        : "error"
                                  }
                                  sx={{ mb: 1 }}
                                />
                                <Stack direction="row" spacing={1}>
                                  <Button
                                    variant="outlined"
                                    color="success"
                                    size="small"
                                    startIcon={<CheckCircle />}
                                    onClick={() =>
                                      sendFeedback(prediction.label, true)
                                    }
                                    disabled={feedbackLoading}
                                  >
                                    Correct
                                  </Button>
                                  <Button
                                    variant="outlined"
                                    color="error"
                                    size="small"
                                    startIcon={<Cancel />}
                                    onClick={() =>
                                      sendFeedback(prediction.label, false)
                                    }
                                    disabled={feedbackLoading}
                                  >
                                    Wrong
                                  </Button>
                                </Stack>
                              </CardContent>
                            </Card>
                          </Grid>
                        ))}
                      </Grid>
                    </Stack>
                  )}
                </Stack>
              </Box>
            </Box>
          )}

          {/* Albums Page */}
          {currentPage === "albums" && (
            <AlbumsPage onSelectAlbum={handleSelectAlbum} />
          )}

          {/* Album Details Page */}
          {currentPage === "album-details" && selectedAlbumId && (
            <AlbumDetailsPage
              albumId={selectedAlbumId}
              albumName={selectedAlbumName || undefined}
              onBack={handleBackToAlbums}
            />
          )}
        </Box>

        {/* Feedback Dialog */}
        <Dialog
          open={dialogOpen}
          onClose={handleDialogClose}
          maxWidth="sm"
          fullWidth
        >
          <DialogTitle>Correct Species Label</DialogTitle>
          <DialogContent sx={{ pt: 2 }}>
            <Typography variant="body2" sx={{ mb: 2 }}>
              The model predicted: <strong>{pendingFeedback?.label}</strong>
            </Typography>
            <Typography variant="body2" sx={{ mb: 2 }}>
              What is the correct species name?
            </Typography>
            <TextField
              autoFocus
              fullWidth
              label="Correct species label"
              value={correctLabel}
              onChange={(e) => setCorrectLabel(e.target.value)}
              placeholder="e.g., Oak, Pine, Birch"
              disabled={feedbackLoading}
            />
          </DialogContent>
          <DialogActions>
            <Button onClick={handleDialogClose} disabled={feedbackLoading}>
              Cancel
            </Button>
            <Button
              onClick={handleDialogSubmit}
              variant="contained"
              disabled={feedbackLoading}
              startIcon={
                feedbackLoading ? <CircularProgress size={20} /> : null
              }
            >
              {feedbackLoading ? "Submitting..." : "Submit Feedback"}
            </Button>
          </DialogActions>
        </Dialog>

        {/* Footer */}
        <Box
          component="footer"
          sx={{
            backgroundColor: "white",
            borderTop: 1,
            borderColor: "grey.200",
            py: 2,
          }}
        >
          <Box sx={{ maxWidth: 1200, mx: "auto", px: 2, textAlign: "center" }}>
            <Typography variant="body2" color="text.secondary">
              PetreeBlitz - Powered by AI for Archaeobotany Research
            </Typography>
          </Box>
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;
