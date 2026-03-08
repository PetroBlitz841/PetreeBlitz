import React from "react";
import { Box, Stack, Typography, Chip } from "@mui/material";
import { Plane, PLANES, Prediction } from "../types";
import ActivePlaneUpload from "./ActivePlaneUpload";
import DemoPlaneUpload from "./DemoPlaneUpload";

interface PlaneSectionProps {
  plane: Plane;
  // shared drag state
  dragOver: Plane | null;
  setDragOver: (plane: Plane | null) => void;
  // active (traverse) props
  fileInputRef: React.RefObject<HTMLInputElement | null>;
  file: File | null;
  imagePreview: string | null;
  loading: boolean;
  error: string | null;
  results: Prediction[];
  onFileSelect: (file: File) => void;
  onIdentify: () => void;
  onCorrect: () => void;
  onWrong: (label: string) => void;
  // demo props
  demoFile: File | null;
  demoPreview: string | null;
  onDemoFileSelect: (plane: Plane, file: File) => void;
}

export default function PlaneSection({
  plane,
  dragOver,
  setDragOver,
  fileInputRef,
  file,
  imagePreview,
  loading,
  error,
  results,
  onFileSelect,
  onIdentify,
  onCorrect,
  onWrong,
  demoFile,
  demoPreview,
  onDemoFileSelect,
}: PlaneSectionProps) {
  const metadata = PLANES[plane];

  return (
    <Box>
      <Stack direction="row" alignItems="center" spacing={1} mb={1}>
        <Typography variant="subtitle1" fontWeight={600}>
          {metadata.label}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          — {metadata.description}
        </Typography>
        {!metadata.active && (
          <Chip
            label="Demo only"
            size="small"
            color="warning"
            variant="outlined"
          />
        )}
      </Stack>

      {metadata.active ? (
        <ActivePlaneUpload
          dragOver={dragOver}
          setDragOver={setDragOver}
          fileInputRef={fileInputRef}
          file={file}
          imagePreview={imagePreview}
          loading={loading}
          error={error}
          results={results}
          onFileSelect={onFileSelect}
          onIdentify={onIdentify}
          onCorrect={onCorrect}
          onWrong={onWrong}
        />
      ) : (
        <DemoPlaneUpload
          plane={plane}
          dragOver={dragOver}
          setDragOver={setDragOver}
          file={demoFile}
          preview={demoPreview}
          onFileSelect={onDemoFileSelect}
        />
      )}
    </Box>
  );
}
