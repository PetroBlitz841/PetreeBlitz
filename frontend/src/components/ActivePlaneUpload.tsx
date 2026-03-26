import { RefObject } from "react";
import {
  Box,
  Stack,
  Button,
  CircularProgress,
  Alert,
  Card,
  CardMedia,
} from "@mui/material";
import UploadCard from "./UploadCard";
import ResultsWithFeatures from "./ResultsWithFeatures";
import {
  Plane,
  Prediction,
  IAWAFeatureResult,
  FeatureSpeciesSupport,
  FeatureCorrection,
} from "../types";

interface ActivePlaneUploadProps {
  dragOver: Plane | null;
  setDragOver: (plane: Plane | null) => void;
  fileInputRef: RefObject<HTMLInputElement | null>;
  file: File | null;
  imagePreview: string | null;
  loading: boolean;
  error: string | null;
  results: Prediction[];
  features: IAWAFeatureResult[];
  featureSupport: FeatureSpeciesSupport;
  corrections: FeatureCorrection[];
  onFileSelect: (file: File) => void;
  onIdentify: () => void;
  onCorrect: (label: string) => void;
  onNewSpecies: () => void;
  onCorrectionsChange: (corrections: FeatureCorrection[]) => void;
}

export default function ActivePlaneUpload({
  dragOver,
  setDragOver,
  fileInputRef,
  file,
  imagePreview,
  loading,
  error,
  results,
  features,
  featureSupport,
  corrections,
  onFileSelect,
  onIdentify,
  onCorrect,
  onNewSpecies,
  onCorrectionsChange,
}: ActivePlaneUploadProps) {
  return (
    <Stack direction="column" spacing={2}>
      <UploadCard
        dragOver={dragOver === "traverse"}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver("traverse");
        }}
        onDragLeave={(e) => {
          e.preventDefault();
          setDragOver(null);
        }}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(null);
          const files = e.dataTransfer.files;
          if (files.length > 0) onFileSelect(files[0]);
        }}
        onClick={() => fileInputRef?.current?.click()}
        fileName={file?.name}
      />
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        style={{ display: "none" }}
        onChange={(e) => {
          if (e.target.files && e.target.files.length > 0)
            onFileSelect(e.target.files[0]);
        }}
      />

      {imagePreview && (
        <Box display="flex" justifyContent="center">
          <Card sx={{ width: "60%" }}>
            <CardMedia
              component="img"
              height="300"
              image={imagePreview}
              alt="Uploaded transverse sample"
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
            onClick={onIdentify}
            disabled={loading}
            startIcon={loading ? <CircularProgress size={20} /> : null}
          >
            {loading ? "Identifying..." : "Identify Tree Species"}
          </Button>
        </Box>
      )}

      {error && <Alert severity="error">{error}</Alert>}

      <ResultsWithFeatures
        results={results}
        features={features}
        featureSupport={featureSupport}
        corrections={corrections}
        feedbackLoading={loading}
        onCorrectionsChange={onCorrectionsChange}
        onCorrect={onCorrect}
        onNewSpecies={onNewSpecies}
      />
    </Stack>
  );
}
