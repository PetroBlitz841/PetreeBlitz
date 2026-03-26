import React from "react";
import { Box, Stack, Alert, Card, CardMedia } from "@mui/material";
import UploadCard from "./UploadCard";
import { Plane, PLANES } from "../../types";

interface DemoPlaneUploadProps {
  plane: Plane;
  dragOver: Plane | null;
  setDragOver: (plane: Plane | null) => void;
  file: File | null;
  preview: string | null;
  onFileSelect: (plane: Plane, file: File) => void;
}

export default function DemoPlaneUpload({
  plane,
  dragOver,
  setDragOver,
  file,
  preview,
  onFileSelect,
}: DemoPlaneUploadProps) {
  const inputRef = React.useRef<HTMLInputElement>(null);
  const metadata = PLANES[plane];

  return (
    <Stack direction="column" spacing={2}>
      <UploadCard
        dragOver={dragOver === plane}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(plane);
        }}
        onDragLeave={(e) => {
          e.preventDefault();
          setDragOver(null);
        }}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(null);
          const files = e.dataTransfer.files;
          if (files.length > 0) onFileSelect(plane, files[0]);
        }}
        onClick={() => inputRef.current?.click()}
        fileName={file?.name}
      />
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        style={{ display: "none" }}
        onChange={(e) => {
          if (e.target.files && e.target.files.length > 0)
            onFileSelect(plane, e.target.files[0]);
        }}
      />

      {preview && (
        <Box display="flex" justifyContent="center">
          <Card sx={{ width: "60%" }}>
            <CardMedia
              component="img"
              height="300"
              image={preview}
              alt={`Uploaded ${metadata.label} sample`}
              sx={{ objectFit: "contain" }}
            />
          </Card>
        </Box>
      )}

      {file && (
        <Alert severity="info">
          <strong>{metadata.label}</strong> identification is not available in
          this demo. Only the Transverse plane is functional.
        </Alert>
      )}
    </Stack>
  );
}
