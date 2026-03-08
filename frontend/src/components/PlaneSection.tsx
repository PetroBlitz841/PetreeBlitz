import { Box, Stack, Typography, Chip } from "@mui/material";
import { Plane, PLANES, Prediction } from "../types";
import ActivePlaneUpload from "./ActivePlaneUpload";
import DemoPlaneUpload from "./DemoPlaneUpload";
import PlaneDiagram from "./PlaneDiagram";
import { RefObject } from "react";

interface PlaneSectionProps {
  plane: Plane;
  dragOver: Plane | null;
  setDragOver: (plane: Plane | null) => void;
  fileInputRef: RefObject<HTMLInputElement | null>;
  file: File | null;
  imagePreview: string | null;
  loading: boolean;
  error: string | null;
  results: Prediction[];
  onFileSelect: (file: File) => void;
  onIdentify: () => void;
  onCorrect: () => void;
  onWrong: (label: string) => void;
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
      <Stack direction="row" alignItems="center" spacing={1.5} mb={1}>
        <PlaneDiagram plane={plane} size={64} />
        <Stack direction="column" spacing={0.25}>
          <Stack direction="row" alignItems="center" spacing={1}>
            <Typography variant="subtitle1" fontWeight={600}>
              {metadata.label}
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
          <Typography variant="body2" color="text.secondary">
            {metadata.description}
          </Typography>
        </Stack>
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
