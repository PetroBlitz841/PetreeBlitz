import React from "react";
import { Box, Stack, Typography, Alert, IconButton } from "@mui/material";
import SettingsDialog from "../components/SettingsDialog";
import FeedbackDialog from "../components/FeedbackDialog";
import PlaneSection from "../components/PlaneSection";
import { useIdentify } from "../hooks/useIdentify";
import { Settings, FeedbackPayload, DEFAULT_SETTINGS, Plane } from "../types";
import { Settings as SettingsIcon } from "@mui/icons-material";
import { usePersistedStorage } from "../hooks/useStorage";

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

  const [settingsOpen, setSettingsOpen] = React.useState(false);
  const [settings, setSettings] = usePersistedStorage<Settings>(
    "identify_settings",
    DEFAULT_SETTINGS,
  );
  const [dragOver, setDragOver] = React.useState<Plane | null>(null);
  const [feedbackOpen, setFeedbackOpen] = React.useState(false);
  const [pendingLabel, setPendingLabel] = React.useState<string | null>(null);
  const [demoFiles, setDemoFiles] = React.useState<
    Partial<Record<Plane, { file: File; preview: string }>>
  >({});

  const handleSettingsOpen = () => setSettingsOpen(true);
  const handleSettingsClose = () => setSettingsOpen(false);
  const handleSettingsSubmit = (newSettings: Settings) => {
    setSettings(newSettings);
    setSettingsOpen(false);
  };

  const handleDemoFileSelect = (plane: Plane, f: File) => {
    const url = URL.createObjectURL(f);
    setDemoFiles((prev) => ({ ...prev, [plane]: { file: f, preview: url } }));
  };

  const handleCorrect = () => {
    if (!sampleId) return;
    const payload: FeedbackPayload = { sample_id: sampleId, was_correct: true };
    sendFeedback(payload);
  };

  const handleWrong = (label: string) => {
    setPendingLabel(label);
    setFeedbackOpen(true);
  };

  const handleFeedbackCancel = () => {
    setFeedbackOpen(false);
    setPendingLabel(null);
  };

  const handleFeedbackSubmit = (maybeLabel?: string) => {
    if (!sampleId || !pendingLabel) return;
    const payload: FeedbackPayload = {
      sample_id: sampleId,
      was_correct: false,
    };
    if (maybeLabel) payload.correct_label = maybeLabel;
    sendFeedback(payload);
    setFeedbackOpen(false);
    setPendingLabel(null);
  };

  // Collect enabled planes from settings
  const enabledPlanes = (
    Object.entries(settings.planes) as [Plane, boolean][]
  ).filter(([, enabled]) => enabled);

  return (
    <Box
      sx={{
        flex: 1,
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        width: "100%",
        px: 2,
        py: 4,
      }}
    >
      {/* Settings button */}
      <Box sx={{ position: "fixed", top: 16, right: 16, zIndex: 10 }}>
        <IconButton onClick={handleSettingsOpen}>
          <SettingsIcon />
        </IconButton>
      </Box>

      <Box sx={{ width: "100%", maxWidth: 900 }}>
        <Stack direction="column" spacing={3}>
          {/* Header */}
          <Stack direction="column" spacing={1}>
            <Typography variant="h4" color="primary" textAlign="center">
              Identify Tree Species from Images
            </Typography>
            <Typography variant="h6" textAlign="center" color="text.secondary">
              Upload a photo of a tree sample and our AI will identify the
              species with confidence scores. Help improve our model by
              providing feedback on predictions.
            </Typography>
          </Stack>

          {/* Per-plane upload sections */}
          {enabledPlanes.length === 0 ? (
            <Alert severity="warning">
              No planes selected. Open Settings to enable at least one cut
              plane.
            </Alert>
          ) : (
            enabledPlanes.map(([plane]) => (
              <PlaneSection
                key={plane}
                plane={plane}
                dragOver={dragOver}
                setDragOver={setDragOver}
                fileInputRef={fileInputRef}
                file={file}
                imagePreview={imagePreview}
                loading={loading}
                error={error}
                results={results}
                onFileSelect={handleFileSelect}
                onIdentify={identify}
                onCorrect={handleCorrect}
                onWrong={handleWrong}
                demoFile={demoFiles[plane]?.file ?? null}
                demoPreview={demoFiles[plane]?.preview ?? null}
                onDemoFileSelect={handleDemoFileSelect}
              />
            ))
          )}
        </Stack>

        {/* Dialogs */}
        <SettingsDialog
          open={settingsOpen}
          initialSettings={settings}
          onCancel={handleSettingsClose}
          onSubmit={handleSettingsSubmit}
        />
        <FeedbackDialog
          open={feedbackOpen}
          predictedLabel={pendingLabel || undefined}
          loading={loading}
          onCancel={handleFeedbackCancel}
          onSubmit={handleFeedbackSubmit}
        />
      </Box>
    </Box>
  );
}
