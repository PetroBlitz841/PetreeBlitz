import { RefObject } from "react";
import { Box, Card, CardMedia } from "@mui/material";
import UploadCard from "./UploadCard";
import { Plane } from "../types";

interface ActivePlaneUploadProps {
  dragOver: Plane | null;
  setDragOver: (plane: Plane | null) => void;
  fileInputRef: RefObject<HTMLInputElement | null>;
  file: File | null;
  imagePreview: string | null;
  onFileSelect: (file: File) => void;
}

export default function ActivePlaneUpload({
  dragOver,
  setDragOver,
  fileInputRef,
  file,
  imagePreview,
  onFileSelect,
}: ActivePlaneUploadProps) {
  return (
    <Box display="flex" flexDirection="column" gap={2}>
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
    </Box>
  );
}
