import React from "react";
import {
  Box,
  Stack,
  Typography,
  Alert,
  IconButton,
  Button,
  CircularProgress,
  Divider,
} from "@mui/material";
import Grid from "@mui/material/Grid";
import SettingsDialog from "../components/identify/SettingsDialog";
import FeedbackDialog from "../components/identify/FeedbackDialog";
import PlaneSection from "../components/identify/PlaneSection";
import ResultsWithFeatures from "../components/identify/ResultsWithFeatures";
import { useIdentify } from "../hooks/useIdentify";
import {
  Settings,
  FeedbackPayload,
  DEFAULT_SETTINGS,
  Plane,
  FeatureCorrection,
} from "../types";
import { Settings as SettingsIcon } from "@mui/icons-material";
import { usePersistedStorage } from "../hooks/useStorage";
import { isTiffFile, convertTiffFileToPng } from "../utils/tiff";

export default function IdentifyPage() {
  const {
    file,
    imagePreview,
    results,
    features,
    featureSupport,
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
  const [featureCorrections, setFeatureCorrections] = React.useState<
    FeatureCorrection[]
  >([]);
  const [demoFiles, setDemoFiles] = React.useState<
    Partial<Record<Plane, { file: File; preview: string }>>
  >({});

  // Reset corrections whenever a new identification runs (results change)
  React.useEffect(() => {
    setFeatureCorrections([]);
  }, [results]);

  const handleSettingsOpen = () => setSettingsOpen(true);
  const handleSettingsClose = () => setSettingsOpen(false);
  const handleSettingsSubmit = (newSettings: Settings) => {
    setSettings(newSettings);
    setSettingsOpen(false);
  };

  const handleDemoFileSelect = async (f: File, plane: Plane) => {
    let preview: string;
    if (isTiffFile(f)) {
      try {
        preview = await convertTiffFileToPng(f);
      } catch {
        preview = URL.createObjectURL(f);
      }
    } else {
      preview = URL.createObjectURL(f);
    }
    setDemoFiles((prev) => ({ ...prev, [plane]: { file: f, preview } }));
  };

  const handleCorrect = (label: string) => {
    if (!sampleId) return;
    const payload: FeedbackPayload = {
      sample_id: sampleId,
      was_correct: true,
      correct_label: label,
      feature_corrections:
        featureCorrections.length > 0 ? featureCorrections : undefined,
    };
    sendFeedback(payload);
  };

  const handleNewSpecies = () => {
    setFeedbackOpen(true);
  };

  const handleFeedbackCancel = () => {
    setFeedbackOpen(false);
  };

  const handleFeedbackSubmit = (speciesName: string) => {
    if (!sampleId) return;
    const payload: FeedbackPayload = {
      sample_id: sampleId,
      was_correct: false,
      correct_label: speciesName,
      feature_corrections:
        featureCorrections.length > 0 ? featureCorrections : undefined,
    };
    sendFeedback(payload);
    setFeedbackOpen(false);
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

      <Box sx={{ width: "100%", maxWidth: 1600, mx: "auto" }}>
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
            <Grid container spacing={3} justifyContent="center">
              {enabledPlanes.map(([plane]) => {
                const isActive = plane === "traverse";
                const planeFile = isActive
                  ? file
                  : (demoFiles[plane]?.file ?? null);
                const planePreview = isActive
                  ? imagePreview
                  : (demoFiles[plane]?.preview ?? null);

                return (
                  <Grid size={{ xs: 16, md: 4 }} key={plane}>
                    <PlaneSection
                      plane={plane}
                      dragOver={dragOver}
                      setDragOver={setDragOver}
                      fileInputRef={isActive ? fileInputRef : undefined}
                      file={planeFile}
                      preview={planePreview}
                      onFileSelect={
                        isActive
                          ? handleFileSelect
                          : (f: File) => handleDemoFileSelect(f, plane)
                      }
                    />
                  </Grid>
                );
              })}
            </Grid>
          )}

          {/* Identification action + results */}
          {file && (
            <>
              <Divider />
              <Box textAlign="center">
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
            </>
          )}

          {error && <Alert severity="error">{error}</Alert>}

          <ResultsWithFeatures
            results={results}
            features={features}
            featureSupport={featureSupport}
            corrections={featureCorrections}
            feedbackLoading={loading}
            onCorrectionsChange={setFeatureCorrections}
            onCorrect={handleCorrect}
            onNewSpecies={handleNewSpecies}
          />
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
          loading={loading}
          featureCorrections={featureCorrections}
          onCancel={handleFeedbackCancel}
          onSubmit={handleFeedbackSubmit}
        />
      </Box>
    </Box>
  );
}
