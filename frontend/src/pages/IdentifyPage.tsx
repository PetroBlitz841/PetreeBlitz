import React from "react";
import { Box, Stack, Typography, Button, CircularProgress, Alert, Card, CardMedia } from "@mui/material";
import UploadCard from "../components/UploadCard";
import ResultsList from "../components/ResultsList";
import FeedbackDialog from "../components/FeedbackDialog";
import { useIdentify } from "../hooks/useIdentify";
import { FeedbackPayload } from "../types";

export default function IdentifyPage() {
  const {
    file,
    imagePreview,
    results,
    loading,
    error,
    sampleId,
    fileInputRef,
    handleFileSelect,
    identify,
    sendFeedback,
  } = useIdentify();

  const [dragOver, setDragOver] = React.useState(false);
  const [dialogOpen, setDialogOpen] = React.useState(false);
  const [pendingLabel, setPendingLabel] = React.useState<string | null>(null);

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
    if (files.length > 0) handleFileSelect(files[0]);
  };

  const handleCorrect = () => {
    if (!sampleId) return;
    const payload: FeedbackPayload = { sample_id: sampleId, was_correct: true };
    sendFeedback(payload);
  };
  const handleWrong = (label: string) => {
    setPendingLabel(label);
    setDialogOpen(true);
  };

  const handleDialogCancel = () => {
    setDialogOpen(false);
    setPendingLabel(null);
  };

  const handleDialogSubmit = (maybeLabel?: string) => {
    if (!sampleId || !pendingLabel) return;
    const payload: FeedbackPayload = { sample_id: sampleId, was_correct: false };
    if (maybeLabel) payload.correct_label = maybeLabel;
    sendFeedback(payload);
    setDialogOpen(false);
    setPendingLabel(null);
  };

  return (
    <Box sx={{ flex: 1, display: "flex", justifyContent: "center", alignItems: "center", width: "100%", px: 2, py: 4 }}>
      <Box sx={{ width: "100%", maxWidth: 900 }}>
        <Stack direction="column" spacing={3}>
          <Stack direction="column" spacing={1}>
            <Typography variant="h4" color="primary" textAlign="center">
              Identify Tree Species from Images
            </Typography>
            <Typography variant="h6" textAlign="center" color="text.secondary">
              Upload a photo of a tree sample and our AI will identify the species with confidence scores. Help improve our model by providing feedback on predictions.
            </Typography>
          </Stack>

          <UploadCard
            dragOver={dragOver}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            fileName={file?.name}
          />
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            style={{ display: "none" }}
            onChange={(e) => { if (e.target.files && e.target.files.length > 0) handleFileSelect(e.target.files[0]); }}
          />

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

          {file && (
            <Box sx={{ textAlign: "center" }}>
              <Button
                variant="contained"
                size="large"
                onClick={identify}
                disabled={loading}
                startIcon={loading ? <CircularProgress size={20} /> : null}
              >
                {loading ? "Identifying..." : "Identify Tree Species"}
              </Button>
            </Box>
          )}

          {error && <Alert severity="error">{error}</Alert>}

          <ResultsList
            results={results}
            feedbackLoading={loading}
            onCorrect={() => handleCorrect()}
            onWrong={handleWrong}
          />
        </Stack>

        <FeedbackDialog
          open={dialogOpen}
          predictedLabel={pendingLabel || undefined}
          loading={loading}
          onCancel={handleDialogCancel}
          onSubmit={handleDialogSubmit}
        />
      </Box>
    </Box>
  );
}
