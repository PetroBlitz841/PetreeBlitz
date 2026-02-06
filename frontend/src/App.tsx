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
} from "@mui/material";
import { CloudUpload, CheckCircle, Cancel } from "@mui/icons-material";

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
  species: string;
  confidence: number;
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [results, setResults] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Handle file selection
  const handleFileSelect = (selectedFile: File) => {
    setFile(selectedFile);
    setError(null);
    setResults([]);
    const reader = new FileReader();
    reader.onload = (e) => {
      setImagePreview(e.target?.result as string);
    };
    reader.readAsDataURL(selectedFile);
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

      const response = await axios.post(
        "http://localhost:8000/identify",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        },
      );

      setResults(response.data.predictions || []);
    } catch (err) {
      setError("Failed to identify tree species. Please try again.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Send feedback
  const sendFeedback = async (species: string, correct: boolean) => {
    try {
      await axios.post("http://localhost:8000/feedback", {
        prediction: species,
        correct,
      });
      // Optionally, show a success message or update UI
    } catch (err) {
      console.error("Failed to send feedback:", err);
    }
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
        </Box>

        {/* Main Content */}
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
                  Upload a photo of a tree sample and our AI will identify the
                  species with confidence scores. Help improve our model by
                  providing feedback on predictions.
                </Typography>
              </Stack>

              {/* Upload Area */}
              <Card
                sx={{
                  border: dragOver ? "2px dashed #4caf50" : "2px dashed #ccc",
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
                  <CloudUpload sx={{ fontSize: 48, color: "primary.main" }} />
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
                    startIcon={loading ? <CircularProgress size={20} /> : null}
                  >
                    {loading ? "Identifying..." : "Identify Tree Species"}
                  </Button>
                </Box>
              )}

              {/* Error Display */}
              {error && <Alert severity="error">{error}</Alert>}

              {/* Results Display */}
              {results.length > 0 && (
                <Stack direction="column" spacing={2} sx={{ width: "100%" }}>
                  <Typography variant="h5">Identification Results</Typography>
                  <Grid container spacing={2}>
                    {results.map((prediction, index) => (
                      <Grid size={{ xs: 12, sm: 4 }} key={index}>
                        <Card>
                          <CardContent>
                            <Typography variant="h6" fontWeight="bold">
                              #{index + 1} {prediction.species}
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
                                  sendFeedback(prediction.species, true)
                                }
                              >
                                Correct
                              </Button>
                              <Button
                                variant="outlined"
                                color="error"
                                size="small"
                                startIcon={<Cancel />}
                                onClick={() =>
                                  sendFeedback(prediction.species, false)
                                }
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
