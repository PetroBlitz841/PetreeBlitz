import React from "react";
import { Box, Alert, Card, CardMedia } from "@mui/material";
import UploadCard from "./UploadCard";
import { Plane, PLANES } from "../../types";

interface PlaneUploadProps {
  plane: Plane;
  dragOver: Plane | null;
  setDragOver: (plane: Plane | null) => void;
  fileInputRef?: React.RefObject<HTMLInputElement | null>;
  file: File | null;
  preview: string | null;
  onFileSelect: (file: File) => void;
}

export default function PlaneUpload({
  plane,
  dragOver,
  setDragOver,
  fileInputRef,
  file,
  preview,
  onFileSelect,
}: PlaneUploadProps) {
  const localRef = React.useRef<HTMLInputElement>(null);
  const inputRef = fileInputRef ?? localRef;
  const metadata = PLANES[plane];

  return (
    <Box display="flex" flexDirection="column" gap={2}>
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
          if (files.length > 0) onFileSelect(files[0]);
        }}
        onClick={() => inputRef.current?.click()}
        fileName={file?.name}
      />
      <input
        ref={inputRef}
        type="file"
        accept="image/*,.tif,.tiff"
        style={{ display: "none" }}
        onChange={(e) => {
          if (e.target.files && e.target.files.length > 0)
            onFileSelect(e.target.files[0]);
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

      {!metadata.active && file && (
        <Alert severity="info">
          <strong>{metadata.label}</strong> identification is not available in
          this demo. Only the Transverse plane is functional.
        </Alert>
      )}
    </Box>
  );
}
